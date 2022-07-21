from queue import deque
import numpy as np
from collections import namedtuple
from ..common.storage import SequenceStorage, BatchSequenceStorage, PlusOneSampler, LambdaSampler, batch_items, merge_batches


def default_samplers(sequence_length):
    return [
        PlusOneSampler(sequence_length),
        LambdaSampler(4, lambda _, get: get(-1)[2] == 0.0),
        LambdaSampler(4, lambda _, get: get(-1)[2] != 0.0)
    ]


class ExperienceReplay(SequenceStorage):
    def __init__(self, size, sequence_length):
        super().__init__(size, samplers=default_samplers(sequence_length))

    def sample_sequence(self):
        return self.sample(0)

    def _choose_rp_sequencer(self):
        if self.count(1) < self.samplers[1].sequence_length:
            return 2

        if self.count(2) < self.samplers[2].sequence_length:
            return 1

        if np.random.randint(2) == 0:
            return 1  # from zero 1/3 probability
        else:
            return 2

    def sample_rp_sequence(self):
        return self.sample(self._choose_rp_sequencer())


class BatchExperienceReplay(BatchSequenceStorage):
    def __init__(self, num_processes, size, sequence_length):
        super().__init__(num_processes, size, samplers=default_samplers(sequence_length))

    def sample_sequence(self):
        return self.sample(0)

    def _num_rp_zeros(self):
        # Probability of selecting zero sequence
        fromzeros = np.random.binomial(len(self.storages), 0.3333)
        zeroenvs = sum(self.counts(1))
        nonzeroenvs = sum(self.counts(2))
        fromzeros = min(fromzeros, zeroenvs)
        fromzeros = max(fromzeros, len(self.storages) - nonzeroenvs)
        return fromzeros

    def sample_rp_sequence(self):
        fromzeros = self._num_rp_zeros()
        sampler1_batch = self.sample(1, batch_size=fromzeros)
        sampler2_batch = self.sample(2, len(self.storages) - fromzeros)
        if sampler1_batch is None or sampler2_batch is None:
            return None

        return
        (sampler1_batch, sampler2_batch)
