#!/bin/env python
import gym
import gym.spaces
import numpy as np
from deep_rl.common.tester import fake_env


def assert_in_space(o, space):
    if isinstance(space, gym.spaces.Tuple):
        if not isinstance(o, tuple):
            raise Exception(
                "Observation was not sampled from the space. Expected class %s, real class was %s" % (tuple, type(o)))
        list(map(assert_in_space, o, space.spaces))
    elif isinstance(space, gym.spaces.Box):
        if not isinstance(o, np.ndarray):
            raise Exception("Observation was not sampled from the space. Expected class %s, real class was %s" %
                            (np.ndarray, type(o)))
        if o.dtype != space.dtype:
            raise Exception("Observation was not sampled from the space - invalid dtype")
        if o.shape != space.shape:
            raise Exception(
                "Observation was not sampled from the space. Expected shape was %s, real shape was %s" % (space.shape, o.shape))


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


if __name__ == "__main__":
    from environment import SingleImageWrapper
    from trainer import create_wrapped_environment
    env = SingleImageWrapper(create_wrapped_environment(id="DeepmindLabCustomHouse-v0"))
    test_environment(env)
    print("environment ok")

