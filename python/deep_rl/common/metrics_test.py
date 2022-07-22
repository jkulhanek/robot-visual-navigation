import unittest
import unittest.mock
from .metrics import DataHandler


class DataHandlerTest(unittest.TestCase):
    def testMetricCreateFile(self):
        handler = DataHandler()
        handler.collect([('m1', 0.2)], 1, mode='train')
        handler.collect([('m1', 0.4)], 4, mode='train')
        handler.collect([('m1', 0.6)], 8, mode='train')

        mocked_open = unittest.mock.mock_open(
            read_data='file contents\nas needed\n')
        with unittest.mock.patch('builtins.open', mocked_open):
            # tests calling your code; the open function will use the mocked_open object
            handler.save('/test/')

        mocked_open.assert_called_once_with('/test/metrics.txt', 'w+')
        pass

    def testMetricWriteContent(self):
        handler = DataHandler()
        handler.collect([('m1', 0.2)], 1, mode='train')
        handler.collect([('m1', 0.4)], 4, mode='train')
        handler.collect([('m1', 0.6)], 8, mode='train')

        mocked_open = unittest.mock.mock_open(
            read_data='file contents\nas needed\n')
        with unittest.mock.patch('builtins.open', mocked_open):
            # tests calling your code; the open function will use the mocked_open object
            handler.save('/test/')

        handle = mocked_open()
        handle.write.assert_called_once_with('m1,3,1,4,8,0.2,0.4,0.6\r\n')
        pass

    def testMetricAppendNextSave(self):
        handler = DataHandler()
        handler.collect([('m1', 0.2)], 1, mode='train')
        handler.collect([('m1', 0.4)], 4, mode='train')
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()):
            handler.save('/test/')

        mocked_open = unittest.mock.mock_open()

        handler.collect([('m1', 0.6)], 8, mode='train')
        with unittest.mock.patch('builtins.open', mocked_open):
            handler.save('/test/')

        mocked_open.assert_called_once_with('/test/metrics.txt', 'a')
        pass

    def testMetricAppendNextContent(self):
        handler = DataHandler()
        handler.collect([('m1', 0.2)], 1, mode='train')
        handler.collect([('m1', 0.4)], 4, mode='train')
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()):
            handler.save('/test/')

        mocked_open = unittest.mock.mock_open()

        handler.collect([('m1', 0.6)], 8, mode='train')
        with unittest.mock.patch('builtins.open', mocked_open):
            handler.save('/test/')

        handle = mocked_open()
        handle.write.assert_called_once_with('m1,1,8,0.6\r\n')
        pass


if __name__ == '__main__':
    unittest.main()
