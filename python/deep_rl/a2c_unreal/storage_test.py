import unittest
import numpy as np
from .storage import BatchExperienceReplay


class StorageTest(unittest.TestCase):

    def testRpZerosOnly(self):
        return
        np.random.seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(np.array([1, 5, 3]), np.array([1, 1, 2]),
                 np.array([0, 0, 1]), np.array([0, 0, 0]))
        s.insert(np.array([2, 6, 3]), np.array([1, 1, 2]),
                 np.array([1, 0, 0]), np.array([0, 0, 0]))
        s.insert(np.array([3, 7, 3]), np.array([1, 1, 2]),
                 np.array([0, 1, 0]), np.array([0, 0, 0]))
        s.insert(np.array([4, 8, 3]), np.array([1, 1, 2]),
                 np.array([0, 0, 0]), np.array([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        np.testing.assert_array_equal(sequence[2], np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]))

    def testRpOnesOnly(self):
        np.random.seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(np.array([1, 5, 3]), np.array([1, 1, 2]),
                 np.array([0, 0, 1]), np.array([0, 0, 0]))
        s.insert(np.array([2, 6, 3]), np.array([1, 1, 2]),
                 np.array([1, 0, 0]), np.array([0, 0, 0]))
        s.insert(np.array([3, 7, 3]), np.array([1, 1, 2]),
                 np.array([0, 1, 0]), np.array([0, 0, 0]))
        s.insert(np.array([4, 8, 3]), np.array([1, 1, 2]),
                 np.array([1, 1, 1]), np.array([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        np.testing.assert_array_equal(sequence[2], np.array(
            [[1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]))

    def testRpNormal(self):
        np.random.seed(1)
        s = BatchExperienceReplay(3, 1000, 4)
        s.insert(np.array([1, 5, 3]), np.array([1, 1, 2]),
                 np.array([0, 0, 1]), np.array([0, 0, 0]))
        s.insert(np.array([2, 6, 3]), np.array([1, 1, 2]),
                 np.array([1, 0, 0]), np.array([0, 0, 0]))
        s.insert(np.array([3, 7, 3]), np.array([1, 1, 2]),
                 np.array([0, 1, 0]), np.array([0, 0, 0]))
        s.insert(np.array([4, 8, 3]), np.array([1, 1, 2]),
                 np.array([1, 1, 0]), np.array([0, 0, 0]))
        sequence = s.sample_rp_sequence()
        np.testing.assert_array_equal(sequence[2], np.array(
            [[1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]]))


if __name__ == '__main__':
    unittest.main()
