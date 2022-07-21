from collections import namedtuple
from .core import RolloutBatch
import numpy as np


def _get_batch_size(observations):
    if isinstance(observations, (tuple, list)):
        return _get_batch_size(observations[0])
    elif isinstance(observations, dict):
        return _get_batch_size(observations[observations.keys()[0]])
    else:
        return observations.shape[0]


def batch_observations(observations):
    if isinstance(observations[0], tuple):
        return tuple(batch_observations(list(map(list, observations))))
    elif isinstance(observations[0], list):
        return list(map(lambda *x: batch_observations(x), *observations))
    elif isinstance(observations[0], dict):
        return {key: batch_observations([o[key] for o in observations]) for key in observations[0].keys()}
    else:
        return np.stack(observations, axis=1)


def print_shape(x):
    if isinstance(x, (tuple, list)):
        return '(' + ", ".join(print_shape(y) for y in x) + ')'
    return str(x.shape)


class RolloutStorage:
    def __init__(self, initial_observations, initial_states=[]):
        self.num_processes = _get_batch_size(initial_observations)

        self._terminals = self._last_terminals = np.zeros(
            shape=(self.num_processes,), dtype=np.bool)
        self._states = self._last_states = initial_states
        self._observations = self._last_observations = initial_observations

        self._batch = []

    def _transform_observation(self, observation):
        if isinstance(observation, np.ndarray):
            if observation.dtype == np.uint8:
                return observation.astype(np.float32) / 255.0
            else:
                return observation.astype(np.float32)
        elif isinstance(observation, list):
            return [self._transform_observation(x) for x in observation]
        elif isinstance(observation, tuple):
            return tuple([self._transform_observation(x) for x in observation])

    @property
    def observations(self):
        return self._transform_observation(self._observations)

    @property
    def terminals(self):
        return self._terminals.astype(np.float32)

    @property
    def masks(self):
        return 1 - self.terminals

    @property
    def states(self):
        return self._states

    def insert(self, observations, actions, rewards, terminals, values, states):
        self._batch.append((self._observations, actions,
                           values, rewards, terminals))
        self._observations = observations
        self._terminals = terminals
        self._states = states

    def batch(self, last_values, gamma):
        # Batch in time dimension
        b_actions, b_values, b_rewards, b_terminals = [
            np.stack([b[i] for b in self._batch], axis=1) for i in range(1, 5)]
        b_observations = batch_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        last_returns = (1.0 - b_terminals[:, -1]) * last_values
        b_returns = np.concatenate(
            [np.zeros_like(b_rewards), np.expand_dims(last_returns, 1)], axis=1)
        for n in reversed(range(len(self._batch))):
            b_returns[:, n] = b_rewards[:, n] + \
                gamma * (1.0 - b_terminals[:, n]) * b_returns[:, n + 1]

        # Compute RNN reset masks
        b_masks = (
            1 - np.concatenate([np.expand_dims(self._last_terminals, 1), b_terminals[:, :-1]], axis=1))
        result = RolloutBatch(
            self._transform_observation(b_observations),
            b_returns[:, :-1].astype(np.float32),
            b_actions,
            b_masks.astype(np.float32),
            self._last_states
        )

        self._last_observations = self._observations
        self._last_states = self._states
        self._last_terminals = self._terminals
        self._batch = []
        return result
