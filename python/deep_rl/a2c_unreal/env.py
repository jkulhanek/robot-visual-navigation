from ..common.env import RewardCollector, TransposeImage, ScaledFloatFrame
from ..common.vec_env import DummyVecEnv, SubprocVecEnv
from .util import UnrealEnvBaseWrapper
import gym


def create_env(num_processes, kwargs):
    def thunk(env):
        env = gym.make(**env)
        env = RewardCollector(env)
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        env = UnrealEnvBaseWrapper(env)
        return env

    return SubprocVecEnv([lambda: thunk(kwargs) for _ in range(num_processes)]), DummyVecEnv([lambda: thunk(kwargs)])
