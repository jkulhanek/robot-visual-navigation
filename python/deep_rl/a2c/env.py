from ..common.vec_env import DummyVecEnv, SubprocVecEnv


def create_vec_envs(thunk, num_processes):
    return SubprocVecEnv([thunk for _ in range(num_processes)]), DummyVecEnv([thunk])


def wrap_agent_env(thunk):
    from ..common.env import ScaledFloatFrame, TransposeImage

    def _thunk():
        env = thunk()
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        return env
    return _thunk
