import gym
import gym.spaces
import numpy as np
from .vec_env import VecEnv, flatten_observations
import tempfile
from .torchsummary import get_shape


class TestingEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, dict()


class TestFinished(Exception):
    def __init__(self):
        super().__init__()


class TestingVecEnv(VecEnv):
    def __init__(self, n_processes, observation_space, action_space):
        super().__init__(n_processes, observation_space, action_space)

    def _sample(self):
        return flatten_observations([self.observation_space.sample() for _ in range(self.num_envs)])

    def reset(self):
        return self._sample()

    def step_async(self, action):
        pass

    def step_wait(self):
        return self._sample(), np.zeros((self.num_envs,), dtype=np.float32), np.zeros((self.num_envs,), dtype=np.bool), [dict() for _ in range(self.num_envs)]


def fake_env(env):
    if hasattr(env, 'step_wait'):
        if hasattr(env, 'venv'):
            original = env.venv
            child, original = fake_env(original)
            env.venv = child
            return env, original
        else:
            return TestingVecEnv(env.num_envs, env.observation_space, env.action_space), env

    else:
        if hasattr(env, 'env'):
            original = env.env
            child, original = fake_env(original)
            env.env = child
            return env, original
        else:
            return TestingEnv(env.observation_space, env.action_space), env


def assert_in_space(o, space):
    if isinstance(space, gym.spaces.Tuple):
        if not isinstance(o, tuple):
            raise Exception(
                "Observation was not sampled from the space. Expected class %s, real class was %s" % (tuple, type(o)))
        list(map(assert_in_space, o, space.spaces))
    elif isinstance(space, gym.spaces.Box):
        if not isinstance(o, np.ndarray):
            raise Exception("Observation was not sampled from the space. Expected class %s, real class was %s" % (
                np.ndarray, type(o)))
        if o.dtype != space.dtype:
            raise Exception(
                "Observation was not sampled from the space - invalid dtype")
        if o.shape != space.shape:
            raise Exception("Observation was not sampled from the space. Expected shape was %s, real shape was %s" % (
                space.shape, o.shape))


def test_environment(env, iterations=30):
    env_faked, original = fake_env(env)
    print('Faked environment: %s' % original.__class__.__name__)
    assert_in_space(env_faked.reset(), env_faked.observation_space)
    assert_in_space(env.reset(), env_faked.observation_space)

    action_space = env_faked.action_space
    for _ in range(iterations):
        action = action_space.sample()
        o2, _, _, _ = env.step(action)
        o, _, _, _ = env_faked.step(action)
        assert_in_space(o, env_faked.observation_space)
        assert_in_space(o2, env_faked.observation_space)


def get_space_shape(space):
    if space.__class__.__name__ == 'Box':
        return space.shape

    if space.__class__.__name__ == 'Tuple':
        return tuple(map(get_space_shape, space.spaces))

    raise Exception('Environment type not supported')


def test_trainer(trainer, iterations=30, allow_gpu=False):
    create_env = trainer.unwrapped.create_env

    def wrap_env(env):
        env, original = fake_env(env)
        print('Faked environment: %s' % original.__class__.__name__)
        return env

    def _create_env(*args, **kwargs):
        env = create_env(*args, **kwargs)
        if hasattr(trainer.unwrapped, 'validation_env'):
            trainer.unwrapped.validation_env = wrap_env(
                trainer.unwrapped.validation_env)

        env = wrap_env(env)
        print('Environment shape is %s' %
              str(get_space_shape(env.observation_space)))
        return env

    assert trainer.__class__.__name__ == 'CompiledTrainer'
    with tempfile.TemporaryDirectory() as tmpdir:

        # Replace create env with dummy env
        trainer.unwrapped.create_env = _create_env

        t = trainer
        while hasattr(t, 'trainer'):
            # Redirect saving to temp directory
            if t.__class__.__name__ == 'SaveWrapper':
                t.model_root_directory = tmpdir

            t = t.trainer
        if allow_gpu is not None:
            trainer.unwrapped.allow_gpu = allow_gpu

        trainer.unwrapped.replay_size = 50
        trainer.unwrapped.preprocess_steps = 10

        process_base = trainer.process

        def process(*args, **kwargs):
            res = process_base(*args, **kwargs)

            # Run 30 iterations of process
            if trainer.unwrapped._global_t > iterations:
                raise TestFinished()
            return res
        trainer.process = process

        try:
            trainer.run()
            trainer.run = lambda *args, **kwargs: print(
                'ERROR: Cannot run tested trainer')
        except TestFinished:
            pass
