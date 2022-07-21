from torch.nn import functional as F
import torch
import os

from deep_rl.common.env import RewardCollector, TransposeImage, ScaledFloatFrame
from deep_rl.a2c_unreal.util import UnrealEnvBaseWrapper
from deep_rl.a2c_unreal.unreal import without_last_item, UnrealAgent, UnrealTrainer
from deep_rl.a2c_unreal.util import autocrop_observations
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.pytorch import to_tensor, KeepTensor, detach_all, pytorch_call
from deep_rl.a2c.a2c import expand_time_dimension
from deep_rl import register_trainer, register_agent
from deep_rl.common.schedules import LinearSchedule, MultistepSchedule
from deep_rl import configuration
from model import VisualNavigationModel as Model
from model import UnrealDualModel, DQNModel

NAME = 'dmhouse'
VALIDATION_PROCESSES = 1


def compute_auxiliary_target(observations, cell_size=4, output_size=None):
    with torch.no_grad():
        observations = autocrop_observations(
            observations, cell_size, output_size=output_size).contiguous()
        obs_shape = observations.size()
        abs_diff = observations.view(-1, *obs_shape[2:])
        avg_abs_diff = F.avg_pool2d(abs_diff, cell_size, stride=cell_size)
        return avg_abs_diff.view(*obs_shape[:2] + avg_abs_diff.size()[1:])


def compute_auxiliary_targets(observations, cell_size, output_size):
    observations = observations[0]
    return tuple(map(lambda x: compute_auxiliary_target(x, cell_size, output_size), observations[:3]))


class AuxiliaryTrainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auxiliary_weight = 0.05

    def sample_training_batch(self):
        values, report = super().sample_training_batch()
        aux_batch = self.replay.sample_sequence() if self.auxiliary_weight > 0.0 else None
        values['auxiliary_batch'] = aux_batch
        return values, report

    def compute_auxiliary_loss(self, model, batch, main_device):
        loss, losses = super().compute_auxiliary_loss(model, batch, main_device)
        auxiliary_batch = batch.get('auxiliary_batch')

        # Compute pixel change gradients
        if not auxiliary_batch is None:
            devconv_loss = self._deconv_loss(
                model, auxiliary_batch, main_device)
            loss += (devconv_loss * self.auxiliary_weight)
            losses['aux_loss'] = devconv_loss.item()

        return loss, losses

    def _deconv_loss(self, model, batch, device):
        observations, _, rewards, _ = batch
        observations = without_last_item(observations)
        masks = torch.ones(rewards.size(), dtype=torch.float32, device=device)
        initial_states = to_tensor(
            self._initial_states(masks.size()[0]), device)
        predictions, _ = model.forward_deconv(
            observations, masks, initial_states)
        targets = compute_auxiliary_targets(
            observations, model.deconv_cell_size, predictions[0].size()[3:])
        loss = 0
        for prediction, target in zip(predictions, targets):
            loss += F.mse_loss(prediction, target)

        return loss


class Trainer(AuxiliaryTrainer):
    def __init__(self, *args, allow_gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.num_steps = 20
        self.gamma = .9
        self.allow_gpu = allow_gpu
        self.learning_rate = LinearSchedule(7e-4, 0, self.max_time_steps)
        self.rp_weight = 1.0
        self.pc_weight = 0.05
        self.vr_weight = 1.0
        self.auxiliary_weight = 0.1
        self.entropy_coefficient = 0.001

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0][0]

    def create_env(self, kwargs):
        env, self.validation_env = create_envs(self.num_processes, kwargs)
        return env

    def create_model(self):
        model = Model(
            self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)
        # model_path = os.path.join(configuration.get('models_path'),'chouse-auxiliary4-supervised', 'weights.pth')
        # print('Loading weights from %s' % model_path)
        # model.load_state_dict(torch.load(model_path))
        return model


@register_trainer('turtlebot', max_time_steps=30e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='TurtleLab-v0',
    has_end_action=True
),
    model_kwargs=dict())
class EndTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = LinearSchedule(5e-4, 0, 60e6)
        self.environment_complexity = MultistepSchedule(3.0, [
            (500000, LinearSchedule(3.0, 36.0, 4000000)),
            (4500000, 100.0),
        ])

    def create_model(self):
        model = Model(
            self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)
        model_path = os.path.join(configuration.get(
            'models_path'), 'dmhouse', 'weights.pth')
        print('Loading weights from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model

    def process(self, *args, **kwargs):
        self.env.call_unwrapped(
            "set_complexity", int(self.environment_complexity))
        a, b, metric_context = super().process(*args, **kwargs)
        metric_context.add_last_value_scalar(
            'environment_complexity', int(self.environment_complexity))
        return a, b, metric_context


@register_trainer('turtlebot-noprior', max_time_steps=30e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='TurtleLab-v0',
    has_end_action=True
),
    model_kwargs=dict())
class EndTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = LinearSchedule(5e-4, 0, 60e6)
        self.environment_complexity = MultistepSchedule(3.0, [
            (500000, LinearSchedule(3.0, 36.0, 4000000)),
            (4500000, 100.0),
        ])

    def create_model(self):
        model = Model(
            self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)
        return model

    def process(self, *args, **kwargs):
        self.env.call_unwrapped(
            "set_complexity", int(self.environment_complexity))
        a, b, metric_context = super().process(*args, **kwargs)
        metric_context.add_last_value_scalar(
            'environment_complexity', int(self.environment_complexity))
        return a, b, metric_context


@register_trainer('turtlebot-unreal', max_time_steps=30e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='TurtleLab-v0',
    has_end_action=True
),
    model_kwargs=dict())
@register_trainer('turtlebot-unreal-noprior', preload=False, max_time_steps=30e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='TurtleLab-v0',
    has_end_action=True
),
    model_kwargs=dict())
class UnrealEndTrainer(Trainer):
    def __init__(self, *args, preload=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = LinearSchedule(5e-4, 0, 60e6)
        self.rp_weight = 1.0
        self.pc_weight = 0.05
        self.vr_weight = 1.0
        self.auxiliary_weight = 0.0
        self.environment_complexity = MultistepSchedule(3.0, [
            (500000, LinearSchedule(3.0, 36.0, 4000000)),
            (4500000, 100.0),
        ])
        self.preload = preload

    def create_model(self):
        model = UnrealDualModel(self.env.action_space.n)
        if self.preload:
            model_path = os.path.join(configuration.get(
                'models_path'), 'dmhouse-unreal', 'weights.pth')
            print('Loading weights from %s' % model_path)
            model.load_state_dict(torch.load(model_path))
        return model

    def create_env(self, kwargs):
        env, self.validation_env = create_envs(
            self.num_processes, kwargs, use_dummy=True)
        return env

    def process(self, *args, **kwargs):
        self.env.call_unwrapped(
            "set_complexity", int(self.environment_complexity))
        a, b, metric_context = super().process(*args, **kwargs)
        metric_context.add_last_value_scalar(
            'environment_complexity', int(self.environment_complexity))
        return a, b, metric_context


@register_trainer('turtlebot-a2c', max_time_steps=30e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='TurtleLab-v0',
    has_end_action=True
),
    model_kwargs=dict())
@register_trainer('turtlebot-a2c-noprior', preload=False, max_time_steps=30e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='TurtleLab-v0',
    has_end_action=True
),
    model_kwargs=dict())
class UnrealEndTrainer(Trainer):
    def __init__(self, *args, preload=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = LinearSchedule(5e-4, 0, 60e6)
        self.rp_weight = 0
        self.pc_weight = 0.0
        self.vr_weight = 0
        self.auxiliary_weight = 0.0
        self.environment_complexity = MultistepSchedule(3.0, [
            (500000, LinearSchedule(3.0, 36.0, 4000000)),
            (4500000, 100.0),
        ])
        self.preload = preload

    def create_model(self):
        model = UnrealDualModel(self.env.action_space.n)
        if self.preload:
            model_path = os.path.join(configuration.get(
                'models_path'), 'dmhouse-a2c', 'weights.pth')
            print('Loading weights from %s' % model_path)
            model.load_state_dict(torch.load(model_path))
        return model

    def create_env(self, kwargs):
        env, self.validation_env = create_envs(
            self.num_processes, kwargs, use_dummy=True)
        return env

    def process(self, *args, **kwargs):
        self.env.call_unwrapped(
            "set_complexity", int(self.environment_complexity))
        a, b, metric_context = super().process(*args, **kwargs)
        metric_context.add_last_value_scalar(
            'environment_complexity', int(self.environment_complexity))
        return a, b, metric_context


@register_trainer('dmhouse-a2c', max_time_steps=10e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='DeepmindLabCustomHouse-v0',
),
    model_kwargs=dict())
class DmhouseA2CTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rp_weight = 0
        self.pc_weight = 0.0
        self.vr_weight = 0
        self.auxiliary_weight = 0.0
        self.gamma = .99
        self.learning_rate = LinearSchedule(7e-4, 0, 40e6)

    def create_model(self):
        model = UnrealDualModel(self.env.action_space.n)
        return model


@register_trainer('dmhouse-unreal', max_time_steps=10e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='DeepmindLabCustomHouse-v0',
),
    model_kwargs=dict())
class DmhouseUnrealTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rp_weight = 1.0
        self.pc_weight = 0.05
        self.vr_weight = 1.0
        self.gamma = .99
        self.auxiliary_weight = 0.0
        self.learning_rate = LinearSchedule(7e-4, 0, 40e6)

    def create_model(self):
        model = UnrealDualModel(self.env.action_space.n)
        return model


@register_trainer('dmhouse', max_time_steps=10e6, validation_period=200, validation_episodes=20,  episode_log_interval=10, saving_period=100000, save=True, env_kwargs=dict(
    id='DeepmindLabCustomHouse-v0',
),
    model_kwargs=dict())
class DmhouseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DmhouseTrainer, self).__init__(*args, **kwargs)
        self.learning_rate = LinearSchedule(7e-4, 0, 40e6)
        self.gamma = 0.99


@register_agent("turtlebot", actions=5)
@register_agent("turtlebot-noprior", actions=5)
@register_agent("dmhouse", actions=5)
class Agent(UnrealAgent):
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        super().__init__(*args, **kwargs)

    def create_model(self, *args):
        model = Model(3, self.actions)
        return model

    def wrap_env(self, env):
        return env

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
                observations = expand_time_dimension(observations)
                masks = torch.ones((1, 1), dtype=torch.float32)

                policy_logits, _, states = model(observations, masks, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                return [action.item()], KeepTensor(detach_all(states))

        self._step = step
        return model

    def act(self, obs):
        if self.states is None:
            self.states = self.model.initial_states(1)
        action, self.states = self._step(obs, self.states)
        return action


@register_agent("dmhouse-a2c", actions=5)
@register_agent("dmhouse-unreal", actions=5)
@register_agent("turtlebot-a2c", actions=5)
@register_agent("turtlebot-unreal", actions=5)
@register_agent("turtlebot-a2c-noprior", actions=5)
@register_agent("turtlebot-unreal-noprior", actions=5)
class UAgent(Agent):
    def create_model(self, *args):
        model = UnrealDualModel(self.actions)
        return model


def create_envs(num_training_processes, env_kwargs, use_dummy=False, wrap=None):
    from environment import create_multiscene

    def wrap_internal(env):
        if not wrap is None:
            env = wrap(env)
        env = RewardCollector(env)
        env = TransposeImage(env)
        env = UnrealEnvBaseWrapper(env)
        return env
    env = create_multiscene(
        num_training_processes, wrap=wrap_internal, use_dummy=use_dummy, **env_kwargs)
    val_env = create_multiscene(
        VALIDATION_PROCESSES, wrap=wrap_internal, use_dummy=use_dummy, **env_kwargs)
    return env, val_env


def create_wrapped_environment(**kwargs):
    import gym
    import environment
    env = gym.make(**kwargs)
    env = RewardCollector(env)
    env = TransposeImage(env)
    env = UnrealEnvBaseWrapper(env)
    return env


def default_args():
    return dict(
        env_kwargs=dict(
            id='TurtleLab-v0',
        ),
        model_kwargs=dict()
    )
