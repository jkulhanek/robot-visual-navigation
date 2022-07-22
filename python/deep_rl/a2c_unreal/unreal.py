from abc import abstractclassmethod
from collections import namedtuple
import tempfile
import numpy as np
import os
import time
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..core import AbstractTrainer, SingleTrainer, AbstractAgent
from ..common import MetricContext
from ..common.torchsummary import minimal_summary
from ..common.pytorch import pytorch_call, to_tensor, to_numpy, KeepTensor, detach_all
from ..a2c.storage import RolloutStorage
from ..a2c.a2c import get_batch_size, expand_time_dimension, A2CAgent
from ..common.schedules import LinearSchedule

from .util import pixel_control_loss, value_loss, reward_prediction_loss
from .storage import BatchExperienceReplay
from .model import UnrealModel

UnrealLoss = namedtuple('UnrealLoss', [
                        'a2c_loss', 'actor_loss', 'value_loss', 'entropy', 'pixel_control_loss'])


def without_last_item(inputs):
    if isinstance(inputs, list):
        return [without_last_item(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(without_last_item(list(inputs)))
    else:
        return inputs[:, :-1]


class UnrealModelBase:
    def __init__(self, max_time_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = 0.99
        self.entropy_coefficient = 0.001
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 0.1

        self.learning_rate = LinearSchedule(7e-4, 1e-10, max_time_steps)

        # Auxiliary config
        self.pc_gamma = 0.9
        self.pc_weight = 0.05

        self.vr_weight = 1.0
        self.rp_weight = 1.0

        # Must be initialized in child class
        self.num_processes = None
        self.num_steps = None
        self.env = None

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

        def extend_batch_dim(batch_shape, space):
            if space.__class__.__name__ == 'Box':
                return (batch_shape,) + space.shape
            elif space.__class__.__name__ == 'Tuple':
                return tuple([extend_batch_dim(batch_shape, x) for x in space.spaces])

        shapes = (extend_batch_dim(batch_shape, self.env.observation_space),
                  batch_shape, get_shape_rec(self._initial_states(self.num_processes)))
        minimal_summary(model, shapes)

    def _loss_a2c(self, model, a2c_batch):
        inputs, returns, actions, masks, states = a2c_batch
        policy_logits, value, _ = model(inputs, masks, states)

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

        return loss, action_loss.detach(), value_loss.detach(), dist_entropy.detach()

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0]

    def _loss_pixel_control(self, model, batch, device):
        observations, actions, rewards, terminals = batch
        masks = torch.ones(rewards.size(), dtype=torch.float32, device=device)
        initial_states = to_tensor(
            self._initial_states(masks.size()[0]), device)
        predictions, _ = model.pixel_control(
            observations, masks, initial_states)
        predictions[:, -1].mul_(1.0 - terminals[:, -
                                2].view(*terminals[:, -2].size(), 1, 1, 1))
        pure_observations = self._get_input_for_pixel_control(observations)
        return pixel_control_loss(pure_observations, actions[:, :-1], predictions, self.pc_gamma, cell_size=model.pc_cell_size)

    def _loss_value_replay(self, model, batch, device):
        observations, actions, rewards, terminals = batch
        masks = torch.ones(rewards.size(), dtype=torch.float32, device=device)
        initial_states = to_tensor(
            self._initial_states(masks.size()[0]), device)
        predictions, _ = model.value_prediction(
            observations, masks, initial_states)
        predictions = predictions.squeeze(-1)
        predictions[:, -1].mul_(1.0 - terminals[:, -2])
        return value_loss(predictions, rewards[:, :-1], self.gamma)

    def _loss_reward_prediction(self, model, batch):
        observations, actions, rewards, terminals = batch
        predictions = model.reward_prediction(without_last_item(observations))
        return reward_prediction_loss(predictions, rewards[:, -1])

    def compute_auxiliary_loss(self, model, batch, main_device):
        loss = 0
        pixel_control_batch = batch.get('pixel_control_batch')
        value_replay_batch = batch.get('value_replay_batch')
        reward_prediction_batch = batch.get('reward_prediction_batch')
        losses = dict()
        # Compute pixel change gradients
        if not pixel_control_batch is None:
            pixel_control_loss = self._loss_pixel_control(
                model, pixel_control_batch, main_device)
            loss += (pixel_control_loss * self.pc_weight)
            losses['pc_loss'] = pixel_control_loss.item()

        # Compute value replay gradients
        if not value_replay_batch is None:
            value_replay_loss = self._loss_value_replay(
                model, value_replay_batch, main_device)
            loss += (value_replay_loss * self.vr_weight)
            losses['vr_loss'] = value_replay_loss.item()

        # Compute reward prediction gradients
        if not reward_prediction_batch is None:
            reward_prediction_loss = self._loss_reward_prediction(
                model, reward_prediction_batch)
            loss += (reward_prediction_loss * self.rp_weight)
            losses['rp_loss'] = reward_prediction_loss.item()

        return loss, losses

    def _build_train(self, model, main_device):
        optimizer = optim.RMSprop(model.parameters(
        ), self.learning_rate, eps=self.rms_epsilon, alpha=self.rms_alpha)

        @pytorch_call(main_device)
        def train(a2c_batch, **batch):
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            optimizer.zero_grad()

            # Compute A2C gradients
            loss, action_loss, value_loss, dist_entropy = self._loss_a2c(
                model, a2c_batch)

            # loss.backward()
            # When there is not enough memory
            # You may need to accumulate gradients after each auxiliary task
            # And to destroy the computation graph to free memory
            # However, if your graph is small and fit the memory
            # You can accumulate loss and evaluate it after accumulating all
            action_loss, value_loss, dist_entropy = action_loss.item(
            ), value_loss.item(), dist_entropy.item()
            auxiliary_loss, losses = self.compute_auxiliary_loss(
                model, batch, main_device)
            loss += auxiliary_loss

            # Optimize
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.max_gradient_norm)
            optimizer.step()

            loss = loss.item()
            return loss, action_loss, value_loss, dist_entropy, losses

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


class UnrealTrainer(SingleTrainer, UnrealModelBase):
    def __init__(self, name, env_kwargs, model_kwargs, max_time_steps, **kwargs):
        super().__init__(max_time_steps=max_time_steps,
                         env_kwargs=env_kwargs, model_kwargs=model_kwargs)
        self.max_time_steps = max_time_steps
        self.name = name
        self.num_steps = 20
        self.num_processes = 8
        self.gamma = 0.99
        self.allow_gpu = True
        self.replay_size = 2000

        self.log_dir = None
        self.win = None

    def _initialize(self, **model_kwargs):
        model = super()._build_graph(self.allow_gpu, **model_kwargs)
        self.rollouts = RolloutStorage(
            self.env.reset(), self._initial_states(self.num_processes))
        self.replay = BatchExperienceReplay(
            self.num_processes, self.replay_size, self.num_steps)
        self._tstart = time.time()
        return model

    def save(self, path):
        super().save(path)
        self._save(path)

    def _finalize(self):
        if self.log_dir is not None:
            self.log_dir.cleanup()

    def create_model(self):
        return UnrealModel(self.env.observation_space.spaces[0].shape[0], self.env.action_space.n)

    def create_env(self, env):
        from .env import create_env
        env, self.validation_env = create_env(self.num_processes, env)
        return env

    def process(self, context, mode='train', **kwargs):
        if not self.replay.full:
            while not self.replay.full:
                self._sample_experience_batch()

            print('Experience replay full')

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
            last_observations = self.rollouts.observations
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
            self.replay.insert(last_observations, actions, rewards, terminals)

        last_values, _ = self._value(
            self.rollouts.observations, self.rollouts.masks, self.rollouts.states)
        batched = self.rollouts.batch(last_values, self.gamma)

        # Prepare next batch starting point
        return batched, (len(finished_episodes[0]),) + finished_episodes

    def sample_training_batch(self):
        a2c_batch, report = self._sample_experience_batch()

        # Add auxiliary batches
        pc_batch = self.replay.sample_sequence() if self.pc_weight > 0.0 else None
        vr_batch = self.replay.sample_sequence() if self.vr_weight > 0.0 else None
        rp_batch = self.replay.sample_rp_sequence() if self.rp_weight > 0.0 else None
        return dict(
            a2c_batch=a2c_batch,
            pixel_control_batch=pc_batch,
            value_replay_batch=vr_batch,
            reward_prediction_batch=rp_batch
        ), report

    def _process_train(self, context, metric_context):
        batch, report = self.sample_training_batch()
        loss, value_loss, action_loss, dist_entropy, losses = self._train(
            **batch)

        fps = int(self._global_t / (time.time() - self._tstart))
        metric_context.add_cummulative('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('action_loss', action_loss)
        metric_context.add_scalar('entropy', dist_entropy)
        metric_context.add_last_value_scalar('fps', fps)
        for key, value in losses.items():
            metric_context.add_scalar(key, value)

        return self.num_steps * self.num_processes, report, metric_context


class UnrealAgent(A2CAgent):
    def wrap_env(self, env):
        from .env import UnrealEnvBaseWrapper
        env = super().wrap_env(env)
        env = UnrealEnvBaseWrapper(env)
        return env
