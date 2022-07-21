import gym
import gym.spaces
from gym.wrappers import TimeLimit
import random
import numpy as np
from deep_rl.common.env import ScaledFloatFrame
from deep_rl.common.vec_env import SubprocVecEnv, DummyVecEnv


def _createImageEnvironment(**kwargs):
    from .image_collection_environment import ImageEnvironment, ImageEnvironmentWrapper
    return ImageEnvironmentWrapper(TimeLimit(ImageEnvironment(**kwargs), 300))


def _createDmhouseEnvironment(**kwargs):
    import dmhouse  # Required to register gym
    _ = dmhouse

    # Conversion from the game units to meters
    # NOTE: in the paper the distance travelled was computed slightly differently
    # leading to (slightly) different results
    game_units_to_meters = 1 / 57.144
    env = gym.make('DMHouse-v1', **kwargs, renderer='software',
                   level="custom/old_house", distance_scale=game_units_to_meters, steps_repeat=4)
    return ScaledFloatFrame(env)


class SingleImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        template = env.observation_space[0].spaces[0]
        shape = (6,) + template.shape[1:]
        self.observation_space = gym.spaces.Box(
            template.low.min(), template.high.max(), shape, template.dtype)

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
            # pseudo-independent random sequences
            env.seed((seed * i * 231893) % 15487469)
            return env
        funcs.append(func)

    if use_dummy:
        return DummyVecEnv(funcs)

    else:
        return SubprocVecEnv(funcs)



gym.register("TurtleLab-v0", entry_point=_createImageEnvironment,
             kwargs=dict(dataset_name='turtle_room'))
gym.register(
    id='DMHouseCustom-v1',
    entry_point=_createDmhouseEnvironment,
    kwargs=dict()
)
