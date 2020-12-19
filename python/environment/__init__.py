import gym
import gym.spaces
from gym.wrappers import TimeLimit
import random
import numpy as np
from deep_rl.common.env import ScaledFloatFrame
from gym.vector import AsyncVectorEnv, SyncVectorEnv


def _createImageEnvironment(**kwargs):
    from .image_collection_environment import ImageEnvironment, ImageEnvironmentWrapper
    return ImageEnvironmentWrapper(TimeLimit(ImageEnvironment(**kwargs), 300))


def _createDmhouseEnvironment(**kwargs):
    import dmhouse
    env = gym.make('DMHouse-v1', **kwargs, renderer='software')
    return ScaledFloatFrame(env)


class SingleImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        template = env.observation_space[0].spaces[0]
        shape = (6,) + template.shape[1:]
        self.observation_space = gym.spaces.Box(template.low.min(), template.high.max(), shape, template.dtype)

    def observation(self, observation):
        observations, last_reward_action = observation
        observations = np.concatenate((observations[0], observations[1]), 0)
        return observations


def create_multiscene(num_processes, wrap=lambda e: e, seed=None, use_dummy=False, **kwargs):
    funcs = []
    if seed is None:
        seed = random.randint(0, 15487469)

    for i in range(num_processes):
        def func():
            import environment
            env = wrap(gym.make(**kwargs))
            env.seed((seed * i * 231893) % 15487469)  # pseudo-independent random sequences
            return env
        funcs.append(func)

    if use_dummy:
        return SyncVectorEnv(funcs)

    else:
        return AsyncVectorEnv(funcs)


gym.register("TurtleLab-v0", entry_point=_createImageEnvironment, kwargs=dict(dataset_name='turtle_room'))
gym.register(
    id='DMHouseCustom-v1',
    entry_point=_createDmhouseEnvironment,
    kwargs=dict()
)
