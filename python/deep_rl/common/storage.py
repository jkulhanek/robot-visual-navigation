import numpy as np


def batch_items(items):
    if isinstance(items[0], tuple):
        return tuple(map(batch_items, zip(*items)))

    elif isinstance(items[0], list):
        return list(map(batch_items, zip(*items)))

    else:
        return np.stack(items)


def split_batched_items(items, axis=0):
    if isinstance(items, np.ndarray):
        return [np.squeeze(x, 0) for x in np.split(items, items.shape[axis], axis)]
    elif isinstance(items, list):
        return list(map(list, zip(*[split_batched_items(x) for x in items])))
    elif isinstance(items, tuple):
        return list(map(tuple, zip(*[split_batched_items(x) for x in items])))
    else:
        raise Exception('Type not supported')


def _merge_batches(batches, axis):
    if isinstance(batches[0], np.ndarray):
        return np.concatenate(batches, axis)

    elif isinstance(batches[0], tuple):
        return tuple([merge_batches(*[x[i] for x in batches], axis=axis) for i in range(len(batches[0]))])

    elif isinstance(batches[0], list):
        return [merge_batches(*[x[i] for x in batches], axis=axis) for i in range(len(batches[0]))]


def merge_batches(*batches, **kwargs):
    axis = kwargs.get('axis', 0)
    batches = [x for x in batches if not x is None and (
        not isinstance(x, list) or len(x) != 0)]
    if len(batches) == 1:
        return batches[0]
    elif len(batches) == 0:
        return None

    return _merge_batches(batches, axis)


class NewSelectionException(Exception):
    pass


class SequenceSampler:
    def __init__(self, sequence_length):
        self._sequence_length = sequence_length

    def is_allowed(self, sequence_length, getter):
        return sequence_length >= self.sequence_length

    @property
    def sequence_length(self):
        return self._sequence_length

    def sample(self, getter, index, size):
        batch = []
        for i in range(index - self.sequence_length + 1, index + 1):
            batch.append(getter(i))

        batch = batch_items(batch)
        return batch


class LambdaSampler(SequenceSampler):
    def __init__(self, sequence_length, function):
        super().__init__(sequence_length)
        self.selector = function

    def is_allowed(self, sequence_length, getter):
        return super().is_allowed(sequence_length, getter) and self.selector(sequence_length, getter)


class PlusOneSampler(SequenceSampler):
    def __init__(self, sequence_length):
        super().__init__(sequence_length)

    def sample(self, getter, index, size):
        if index == size - 1:
            # Cannot allow the last index to be selected
            raise NewSelectionException()

        batch = []
        for i in range(index - self.sequence_length + 1, index + 1):
            batch.append(getter(i))

        batch.append(getter(index + 1))

        batch = batch_items(batch)
        return batch


class SequenceStorage:
    def __len__(self):
        return len(self.storage)

    def __init__(self, size, samplers=[]):
        self.samplers = samplers

        self.size = size
        self.storage = []

        self.selector_data = np.zeros(
            (self.size, len(self.samplers),), dtype=np.bool)
        self.selector_lengths = [0 for _ in range(len(self.samplers))]

        self.tail = 0
        self.episode_length = 0

    def remove_samplers(self, index):
        for i, sampler in enumerate(self.samplers):
            positive = self.selector_data[index, i]
            self.selector_data[index, i] = False
            if positive:
                self.selector_lengths[i] -= 1

            # Remove items with shorted sequences
            if self.selector_data[(index + sampler.sequence_length - 1) % self.size, i]:
                self.selector_data[(
                    index + sampler.sequence_length - 1) % self.size, i] = False
                self.selector_lengths[i] -= 1

    def insert_samplers(self, index, ctx):
        for i, sampler in enumerate(self.samplers):
            positive = sampler.is_allowed(*ctx)
            self.selector_data[index, i] = positive

            if positive:
                self.selector_lengths[i] += 1

    def __getitem__(self, index):
        position = (
            self.tail + index) % self.size if len(self) == self.size else index
        return self.storage[position]

    def insert(self, observation, action, reward, terminal):
        row = (observation, action, reward, terminal)
        self.episode_length += 1
        ctx = (self.episode_length, lambda i: self[i])

        if len(self) == self.size:
            # Update samplers
            self.remove_samplers(self.tail)

            self.storage[self.tail] = row
            old_tail = self.tail
            self.tail = (self.tail + 1) % self.size

            self.insert_samplers(old_tail, ctx)
        else:
            self.storage.append(row)
            self.insert_samplers(len(self) - 1, ctx)

        if terminal:
            self.episode_length = 0

    def count(self, sampler):
        return self.selector_lengths[sampler]

    def sample(self, sampler):
        result = None
        trials = 0
        while trials < 200 and result is None:
            try:
                sampler_obj = self.samplers[sampler]
                index = np.random.choice(
                    np.where(self.selector_data[:, sampler])[0])
                result = sampler_obj.sample(
                    lambda i: self[i], (index - self.tail) % self.size, len(self.storage))
            except NewSelectionException:
                pass

            trials += 1

        return result

    @property
    def full(self):
        return len(self.storage) == self.size


class BatchSequenceStorage:
    def __init__(self, num_storages, single_size, samplers=[]):
        self.samplers = samplers
        self.storages = [SequenceStorage(
            single_size, samplers=samplers) for _ in range(num_storages)]

    def insert(self, observations, actions, rewards, terminals):
        batch = (observations, actions, rewards, terminals)
        rows = split_batched_items(batch, axis=0)
        for storage, row in zip(self.storages, rows):
            storage.insert(*row)

    @property
    def full(self):
        return all([x.full for x in self.storages])

    def counts(self, sequencer):
        return [x.count(sequencer) for x in self.storages]

    def sample(self, sampler, batch_size=None):
        if batch_size is None:
            batch_size = len(self.storages)

        if batch_size == 0:
            return []

        probs = np.array([x.selector_lengths[sampler]
                         for x in self.storages], dtype=np.float32)
        if np.sum(probs) == 0:
            print("WARNING: Could not sample data from storage")
            return None

        probs = probs / np.sum(probs)
        selected_sources = np.random.choice(
            np.arange(len(self.storages)), size=(batch_size,), p=probs)

        sequences = []
        for source in selected_sources:
            sd = self.storages[source].sample(sampler)
            if sd is None:
                return None

            sequences.append(sd)

        return batch_items(sequences)
