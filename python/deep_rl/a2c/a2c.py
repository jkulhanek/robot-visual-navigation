from abc import abstractclassmethod
from collections import namedtuple
import tempfile
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .env import wrap_agent_env
from ..core import AbstractTrainer, SingleTrainer, AbstractAgent
from ..common.env import VecTransposeImage, make_vec_envs
from ..common import MetricContext
from ..common.torchsummary import minimal_summary

from .model import TimeDistributedConv
from .storage import RolloutStorage
from .core import pytorch_call, to_tensor, to_numpy, KeepTensor, detach_all
from ..configuration import configuration


def get_batch_size(inputs):
    if isinstance(inputs, (list, tuple)):
        return get_batch_size(inputs[0])
    return inputs.size()[0]


def expand_time_dimension(inputs):
    if isinstance(inputs, list):
        return [expand_time_dimension(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_dimension(list(inputs)))
    else:
        batch_size = inputs.size()[0]
        return inputs.view(batch_size, 1, *inputs.size()[1:])


def expand_time_and_batch_dimensions(inputs):
    if isinstance(inputs, list):
        return [expand_time_and_batch_dimensions(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_and_batch_dimensions(list(inputs)))
    else:
        return inputs.unsqueeze(0).unsqueeze(0)


class A2CModel:
    def __init__(self, max_time_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.data_parallel = True
        self.learning_rate = 7e-4

        def not_initialized(*args, **kwargs):
            raise Exception('Not initialized')
        self._train = self._step = self._value = not_initialized

    @abstractclassmethod
    def create_model(self, **kwargs):
        pass

    def show_summary(self, model):
        batch_shape = (self.num_processes, self.num_steps)

        def get_shape_rec(shapes):
            if isinstance(shapes, tuple):
                return tuple(get_shape_rec(list(shapes)))
            elif isinstance(shapes, list):
                return [get_shape_rec(x) for x in shapes]
            else:
                return shapes.size()

        shapes = (batch_shape + self.env.observation_space.shape, batch_shape,
                  get_shape_rec(self._initial_states(self.num_processes)))
        minimal_summary(model, shapes)

    def _build_train(self, model, main_device):
        optimizer = optim.RMSprop(model.parameters(
        ), self.learning_rate, eps=self.rms_epsilon, alpha=self.rms_alpha)

        @pytorch_call(main_device)
        def train(observations, returns, actions, masks, states=[]):
            policy_logits, value, _ = model(observations, masks, states)

            dist = torch.distributions.Categorical(logits=policy_logits)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()

            # Compute losses
            advantages = returns - value.squeeze(-1)
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
            loss = value_loss * self.value_coefficient + \
                action_loss - \
                dist_entropy * self.entropy_coefficient

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.max_gradient_norm)
            optimizer.step()

            return loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item()

        return train

    def _build_graph(self, allow_gpu, **model_kwargs):
        model = self.create_model(**model_kwargs)
        if hasattr(model, 'initial_states'):
            self._initial_states = getattr(model, 'initial_states')
        else:
            self._initial_states = lambda _: []

        # Show summary
        self.show_summary(model)

        cuda_devices = torch.cuda.device_count()
        if cuda_devices == 0 or not allow_gpu:
            print('Using CPU only')
            main_device = torch.device('cpu')
            def get_state_dict(): return model.state_dict()
        elif cuda_devices > 1 and self.data_parallel:
            print('Using %s GPUs' % cuda_devices)
            main_device = torch.device('cuda:0')
            model = nn.DataParallel(model, output_device=main_device)
            model = model.to(main_device)
            def get_state_dict(): return model.module.state_dict()
        else:
            print('Using single GPU')
            main_device = torch.device('cuda:0')
            model = model.to(main_device)
            def get_state_dict(): return model.state_dict()

        model.train()

        # Build train and act functions
        self._train = self._build_train(model, main_device)

        @pytorch_call(main_device)
        def step(observations, masks, states):
            with torch.no_grad():
                batch_size = get_batch_size(observations)
                observations = expand_time_dimension(observations)
                masks = masks.view(batch_size, 1)

                policy_logits, value, states = model(
                    observations, masks, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                action_log_probs = dist.log_prob(action)
                return action.squeeze(1).detach(), value.squeeze(1).squeeze(-1).detach(), action_log_probs.squeeze(1).detach(), KeepTensor(detach_all(states))

        @pytorch_call(main_device)
        def value(observations, masks, states):
            with torch.no_grad():
                batch_size = get_batch_size(observations)
                observations = expand_time_dimension(observations)
                masks = masks.view(batch_size, 1)

                _, value, states = model(observations, masks, states)
                return value.squeeze(1).squeeze(-1).detach(), KeepTensor(detach_all(states))

        self._step = step
        self._value = value
        self._save = lambda path: torch.save(
            get_state_dict(), os.path.join(path, 'weights.pth'))
        self.main_device = main_device
        return model


class A2CTrainer(SingleTrainer, A2CModel):
    def __init__(self, name, env_kwargs, model_kwargs, max_time_steps, **kwargs):
        super().__init__(max_time_steps=max_time_steps,
                         env_kwargs=env_kwargs, model_kwargs=model_kwargs)
        self.max_time_steps = max_time_steps
        self.name = name
        self.num_steps = 5
        self.num_processes = 16
        self.gamma = 0.99
        self.allow_gpu = True

        self.log_dir = None
        self.win = None

    def _initialize(self, **model_kwargs):
        model = super()._build_graph(self.allow_gpu, **model_kwargs)
        self._tstart = time.time()
        self.rollouts = RolloutStorage(
            self.env.reset(), self._initial_states(self.num_processes))
        return model

    def save(self, path):
        super().save(path)
        self._save(path)

    def _finalize(self):
        if self.log_dir is not None:
            self.log_dir.cleanup()

    def create_env(self, env):
        self.log_dir = tempfile.TemporaryDirectory()

        seed = 1
        self.validation_env = make_vec_envs(
            env, seed, 1, self.gamma, self.log_dir.name, None, allow_early_resets=True)
        if len(self.validation_env.observation_space.shape) == 4:
            self.validation_env = VecTransposeImage(self.validation_env)

        envs = make_vec_envs(env, seed + 1, self.num_processes,
                             self.gamma, self.log_dir.name, None, False)

        if len(envs.observation_space.shape) == 4:
            envs = VecTransposeImage(envs)

        return envs

    def process(self, context, mode='train', **kwargs):
        metric_context = MetricContext()
        if mode == 'train':
            return self._process_train(context, metric_context)
        elif mode == 'validation':
            return self._process_validation(metric_context)
        else:
            raise Exception('Mode not supported')

    def _process_validation(self, metric_context):
        done = False
        states = self._initial_states(1)
        ep_reward = 0.0
        ep_length = 0
        n_steps = 0
        observations = self.validation_env.reset()
        while not done:
            action, _, _, states = self._step(
                observations, np.ones((1, 1), dtype=np.float32), states)
            observations, reward, done, infos = self.validation_env.step(
                action)
            done = done[0]
            info = infos[0]

            if 'episode' in info.keys():
                ep_length = info['episode']['l']
                ep_reward = info['episode']['r']
            n_steps += 1

        return n_steps, (ep_length, ep_reward), metric_context

    def _sample_experience_batch(self):
        finished_episodes = ([], [])
        for _ in range(self.num_steps):
            actions, values, action_log_prob, states = self._step(
                self.rollouts.observations, self.rollouts.masks, self.rollouts.states)

            # Take actions in env and look the results
            observations, rewards, terminals, infos = self.env.step(actions)

            # Collect true rewards
            for info in infos:
                if 'episode' in info.keys():
                    finished_episodes[0].append(info['episode']['l'])
                    finished_episodes[1].append(info['episode']['r'])

            self.rollouts.insert(observations, actions,
                                 rewards, terminals, values, states)

        last_values, _ = self._value(
            self.rollouts.observations, self.rollouts.masks, self.rollouts.states)
        batched = self.rollouts.batch(last_values, self.gamma)

        # Prepare next batch starting point
        return batched, (len(finished_episodes[0]),) + finished_episodes

    def _process_train(self, context, metric_context):
        batch, report = self._sample_experience_batch()
        loss, value_loss, action_loss, dist_entropy = self._train(*batch)

        fps = int(self._global_t / (time.time() - self._tstart))
        metric_context.add_cummulative('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('action_loss', action_loss)
        metric_context.add_scalar('entropy', dist_entropy)
        metric_context.add_last_value_scalar('fps', fps)
        return self.num_steps * self.num_processes, report, metric_context


class A2CAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = self._initialize()
        self.states = None

    @abstractclassmethod
    def create_model(self):
        pass

    def _initialize(self):
        checkpoint_dir = configuration.get('models_path')

        path = os.path.join(checkpoint_dir, self.name, 'weights.pth')
        model = self.create_model()
        model.load_state_dict(torch.load(
            path, map_location=torch.device('cpu')))
        model.eval()

        @pytorch_call(torch.device('cpu'))
        def step(observations, states):
            with torch.no_grad():
                batch_size = get_batch_size(observations)
                observations = expand_time_and_batch_dimensions(observations)
                masks = torch.ones((1, 1), dtype=torch.float32)

                policy_logits, _, states = model(observations, masks, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                return action.item(), KeepTensor(detach_all(states))

        self._step = step
        return model

    def wrap_env(self, env):
        def _thunk():
            return env

        return wrap_agent_env(_thunk)()

    def reset_state(self):
        self.states = None

    def act(self, state):
        action, self.states = self._step(state, self.states)
        return action
