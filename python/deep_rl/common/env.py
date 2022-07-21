import os
import time
import gym
from gym import spaces
import gym.wrappers
import gym.vector
import numpy as np
import torch
from gym.spaces.box import Box
from copy import copy

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

try:
    import environments
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets):
    env_id = env_id
    if isinstance(env_id, dict):
        allow_early_resets = env_id.get(
            'allow_early_resets', allow_early_resets)
        env_id = env_id.get('id')

    def _thunk():
        if callable(env_id):
            env = env_id()
        elif env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = gym.make(env_id)
            env = gym.wrappers.AtariPreprocessing(env)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            # env = gym.wrappers.Monitor(env, os.path.join(log_dir, str(rank)), force=True)
            pass

        if is_atari:
            if len(env.observation_space.shape) == 3:
                # TODO: implement same preprocessing as in deepmind
                # env = wrap_deepmind(env)
                pass

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep, allow_early_resets, num_frame_stack=None):
    if isinstance(env_name, dict):
        if num_frame_stack is None and 'num_frame_stack' in env_name:
            num_frame_stack = env_name.get('num_frame_stack')

    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = gym.vector.AsyncVectorEnv(envs)
    else:
        envs = gym.vector.SyncVectorEnv(envs)

    if num_frame_stack is not None:
        if num_frame_stack != 1:
            envs = VecFrameStack(envs, num_frame_stack)
    elif len(envs.observation_space.shape) == 3:
        envs = VecFrameStack(envs, 4)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, %s, must be dim3" % str(op)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = self.transpose_space(self.observation_space)

    def transpose_space(self, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3:
                obs_shape = space.shape
                return Box(
                    space.low[0, 0, 0],
                    space.high[0, 0, 0],
                    [
                        obs_shape[self.op[0]],
                        obs_shape[self.op[1]],
                        obs_shape[self.op[2]]],
                    dtype=space.dtype)
            return space
        elif space.__class__.__name__ == 'Tuple':
            return gym.spaces.Tuple(tuple(map(self.transpose_space, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def transpose_observation(self, ob, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3:
                return ob if ob is None else np.transpose(ob, axes=self.op)
            return ob
        elif space.__class__.__name__ == 'Tuple':
            return tuple(map(lambda x: self.transpose_observation(*x), zip(ob, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def observation(self, ob):
        return self.transpose_observation(ob, self.env.observation_space)


class VecTransposeImage(gym.vector.vector_env.VectorEnvWrapper):
    def __init__(self, venv, transpose=[2, 0, 1]):
        if venv.observation_space.__class__.__name__ != 'Box':
            raise Exception('Env type %s is not supported' %
                            venv.__class__.__name__)

        super().__init__(venv)
        self._transpose = (0,) + tuple([1 + x for x in transpose])
        obs_space = copy(venv.observation_space)
        obs_space.shape = tuple([obs_space.shape[i] for i in self._transpose])
        self.observation_space = obs_space
        self.action_space = venv.action_space

    def reset_wait(self):
        obs = self.env.reset_wait()
        obs = np.transpose(obs, self._transpose)
        return obs

    def reset_async(self):
        self.env.reset_async()

    def close_extras(self, **kwargs):
        self.env.close_extras(**kwargs)

    def close(self, **kwargs):
        self.env.close(**kwargs)

    def seed(self, *args, **kwargs):
        self.env.seed(*args, **kwargs)

    def step_async(self, actions):
        self.env.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.env.step_wait()
        obs = np.transpose(obs, self._transpose)
        return obs, reward, done, info


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = self.transform_space(env.observation_space)

    def transform_space(self, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3 and space.dtype == np.uint8:
                return gym.spaces.Box(low=0.0, high=1.0, shape=space.shape, dtype=np.float32)
            return space
        elif space.__class__.__name__ == 'Tuple':
            return gym.spaces.Tuple(tuple(map(self.transform_space, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def transform_observation(self, ob, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3 and space.dtype == np.uint8:
                return ob if ob is None else np.array(ob).astype(np.float32) / 255.0
            return ob
        elif space.__class__.__name__ == 'Tuple':
            return tuple(map(lambda x: self.transform_observation(*x), zip(ob, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def observation(self, ob):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        if ob is None:
            return None

        return self.transform_observation(ob, self.env.observation_space)


class VecFrameStack(gym.vector.vector_env.VectorEnvWrapper):
    def __init__(self, venv, nstack):
        super().__init__(self, venv)
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        self.observation_space = observation_space

    def step_wait(self):
        obs, rews, news, infos = self.env.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset_wait(self):
        obs = self.env.reset_wait()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs


class RewardCollector(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.rewards = None

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        self.rewards = []

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        assert isinstance(info, dict)

        self.rewards.append(rew)
        info['reward'] = rew
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen}
            info['episode'] = epinfo
