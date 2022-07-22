import unittest
import numpy as np
from .storage import SequenceStorage as ExperienceReplay, SequenceSampler, BatchSequenceStorage, LambdaSampler, PlusOneSampler, merge_batches


class SequenceStorageTest(unittest.TestCase):
    def assertNumpyArrayEqual(self, a1, a2, msg='Arrays must be equal'):
        if not np.array_equal(a1, a2):
            self.fail(msg=f"{a1} != {a2} : " + msg)

    def testShouldStoreAll(self):
        replay = ExperienceReplay(4, samplers=(SequenceSampler(2),))
        replay.insert(1, 0, 0.0, False)
        replay.insert(5, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay[0][0], 2)
        self.assertEqual(replay[1][0], 4)
        self.assertEqual(replay[2][0], 6)
        self.assertEqual(replay[3][0], 7)

    def testNegativeIndex(self):
        replay = ExperienceReplay(4, samplers=(SequenceSampler(2),))
        replay.insert(1, 0, 0.0, False)
        replay.insert(5, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay[-4][0], 2)
        self.assertEqual(replay[-3][0], 4)
        self.assertEqual(replay[-2][0], 6)
        self.assertEqual(replay[-1][0], 7)

    def testLength(self):
        replay = ExperienceReplay(4, samplers=(SequenceSampler(2),))
        self.assertEqual(len(replay), 0)
        replay.insert(1, 0, 0.0, False)
        self.assertEqual(len(replay), 1)
        replay.insert(2, 0, 0.0, False)
        self.assertEqual(len(replay), 2)
        replay.insert(4, 0, 0.0, False)
        self.assertEqual(len(replay), 3)
        replay.insert(6, 0, 0.0, False)
        self.assertEqual(len(replay), 4)
        replay.insert(7, 0, 0.0, False)
        self.assertEqual(len(replay), 4)

    def testSamplerStats(self):
        replay = ExperienceReplay(4, samplers=(
            LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay.selector_lengths[0], 2)

    def testSamplerStatsRemove(self):
        replay = ExperienceReplay(4, samplers=(
            LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [
                                   False, False, False, False])
        replay.insert(2, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [
                                   False, True, False, False])
        replay.insert(4, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [
                                   False, True, True, False])
        replay.insert(6, 0, 0.0, False)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [
                                   False, True, True, True])
        replay.insert(7, 0, 0.0, False)

        self.assertEqual(replay.selector_lengths[0], 2)
        self.assertNumpyArrayEqual(replay.selector_data[:, 0], [
                                   False, False, True, True])

    def testSamplingWithEpisodeEnd(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, samplers=(
            LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, True)
        replay.insert(7, 0, 0.0, False)

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 2)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([6]))

    def testResampling(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, samplers=(
            LambdaSampler(2, lambda _, get: get(-1)[0] % 2 == 0),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, False)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        toBeSampled = set([4, 6])
        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 2)

        self.assertEqual(len(toBeSampled - wasSampled),
                         0, 'something was not sampled')
        self.assertEqual(len(wasSampled), len(toBeSampled),
                         'something was not supposed to be sampled')
        self.assertSetEqual(wasFirst, set([2, 4]))

    def testPlusOneSampling(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, samplers=(PlusOneSampler(2),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, True)
        replay.insert(7, 0, 0.0, False)

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][-1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 3)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([7]))

    def testPlusOneResampling(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, samplers=(PlusOneSampler(2),))

        replay.insert(6, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)
        replay.insert(4, 0, 0.0, False)
        replay.insert(6, 0, 0.0, False)
        replay.insert(7, 0, 0.0, False)

        wasSampled = set()
        wasFirst = set()
        for _ in range(100):
            batch = replay.sample(0)
            wasSampled.add(batch[0][-1])
            wasFirst.add(batch[0][0])
            self.assertEqual(batch[0].shape[0], 3)

        self.assertSetEqual(wasFirst, set([4]))
        self.assertSetEqual(wasSampled, set([7]))

    def testPlusOneShortMemory(self):
        import numpy
        numpy.random.seed(1)

        replay = ExperienceReplay(4, samplers=(PlusOneSampler(2),))

        replay.insert(1, 0, 0.0, False)
        replay.insert(2, 0, 0.0, True)

        for _ in range(100):
            batch = replay.sample(0)
            self.assertIsNone(batch)


class BatchSequenceStorageTest(unittest.TestCase):
    def testStore(self):
        replay = BatchSequenceStorage(2, 4, samplers=[SequenceSampler(2)])

        replay.insert(np.array([1, 2]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))
        replay.insert(np.array([3, 4]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))
        replay.insert(np.array([5, 6]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))
        replay.insert(np.array([7, 8]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))

    def testSampleShape(self):
        replay = BatchSequenceStorage(2, 4, samplers=[SequenceSampler(2)])
        replay.insert(np.array([1, 2]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))
        replay.insert(np.array([3, 4]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))
        replay.insert(np.array([5, 6]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))
        replay.insert(np.array([7, 8]), np.array([1, 1]), np.array(
            [1.0, 1.0]), np.array([False, False]))

        sample = replay.sample(0, batch_size=3)
        self.assertEqual(sample[0].shape, (3, 2,))
        self.assertEqual(sample[1].shape, (3, 2,))
        self.assertEqual(sample[2].shape, (3, 2,))
        self.assertEqual(sample[3].shape, (3, 2,))


class StorageUtilTest(unittest.TestCase):
    def testMergeBatches(self):
        batch1 = (np.ones((2, 5)), [np.zeros((2, 7)), np.ones((2,))])
        batch2 = (np.ones((3, 5)), [np.zeros((3, 7)), np.ones((3,))])
        merges = merge_batches(batch1, batch2)

        self.assertIsInstance(merges, tuple)
        self.assertIsInstance(merges[1], list)
        self.assertIsInstance(merges[0], np.ndarray)

        self.assertTupleEqual(merges[0].shape, (5, 5))
        self.assertTupleEqual(merges[1][0].shape, (5, 7))
        self.assertTupleEqual(merges[1][1].shape, (5,))

    def testZeroBatch(self):
        batch1 = (np.ones((2, 5)), [np.zeros((2, 7)), np.ones((2,))])
        batch2 = []
        merges = merge_batches(batch1, batch2)

        self.assertIsInstance(merges, tuple)
        self.assertIsInstance(merges[1], list)
        self.assertIsInstance(merges[0], np.ndarray)

        self.assertTupleEqual(merges[0].shape, (2, 5))
        self.assertTupleEqual(merges[1][0].shape, (2, 7))
        self.assertTupleEqual(merges[1][1].shape, (2,))


if __name__ == '__main__':
    unittest.main()
