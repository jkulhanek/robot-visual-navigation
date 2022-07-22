import unittest
from .core import MetricContext


class MetricContextTest(unittest.TestCase):
    def testPickableMetricContext(self):
        import pickle
        ctx = MetricContext()

        class MyFile(object):
            def __init__(self):
                self.data = []

            def write(self, stuff):
                self.data.append(stuff)

        pickle.dump(ctx, MyFile())


if __name__ == '__main__':
    unittest.main()
