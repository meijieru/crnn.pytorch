#!/usr/bin/python
# encoding: utf-8

import sys
import unittest
import torch
from torch.autograd import Variable
import collections
origin_path = sys.path
sys.path.append("..")
import utils
sys.path = origin_path


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return a.equal(b)
    elif isinstance(a, str):
        return a == b
    elif isinstance(a, collections.Iterable):
        res = True
        for (x, y) in zip(a, b):
            res = res & equal(x, y)
        return res
    else:
        return a == b


class utilsTestCase(unittest.TestCase):

    def checkConverter(self):
        encoder = utils.strLabelConverter('abcdefghijklmnopqrstuvwxyz')

        # Encode
        # trivial mode
        result = encoder.encode('efa')
        target = (torch.IntTensor([5, 6, 1]), torch.IntTensor([3]))
        self.assertTrue(equal(result, target))

        # batch mode
        result = encoder.encode(['efa', 'ab'])
        target = (torch.IntTensor([5, 6, 1, 1, 2]), torch.IntTensor([3, 2]))
        self.assertTrue(equal(result, target))

        # Decode
        # trivial mode
        result = encoder.decode(
            torch.IntTensor([5, 6, 1]), torch.IntTensor([3]))
        target = 'efa'
        self.assertTrue(equal(result, target))

        # replicate mode
        result = encoder.decode(
            torch.IntTensor([5, 5, 0, 1]), torch.IntTensor([4]))
        target = 'ea'
        self.assertTrue(equal(result, target))

        # raise AssertionError
        def f():
            result = encoder.decode(
                torch.IntTensor([5, 5, 0, 1]), torch.IntTensor([3]))
        self.assertRaises(AssertionError, f)

        # batch mode
        result = encoder.decode(
            torch.IntTensor([5, 6, 1, 1, 2]), torch.IntTensor([3, 2]))
        target = ['efa', 'ab']
        self.assertTrue(equal(result, target))

    def checkOneHot(self):
        v = torch.LongTensor([1, 2, 1, 2, 0])
        v_length = torch.LongTensor([2, 3])
        v_onehot = utils.oneHot(v, v_length, 4)
        target = torch.FloatTensor([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                                    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]])
        assert target.equal(v_onehot)

    def checkAverager(self):
        acc = utils.averager()
        acc.add(Variable(torch.Tensor([1, 2])))
        acc.add(Variable(torch.Tensor([[5, 6]])))
        assert acc.val() == 3.5

        acc = utils.averager()
        acc.add(torch.Tensor([1, 2]))
        acc.add(torch.Tensor([[5, 6]]))
        assert acc.val() == 3.5

    def checkAssureRatio(self):
        img = torch.Tensor([[1], [3]]).view(1, 1, 2, 1)
        img = Variable(img)
        img = utils.assureRatio(img)
        assert torch.Size([1, 1, 2, 2]) == img.size()


def _suite():
    suite = unittest.TestSuite()
    suite.addTest(utilsTestCase("checkConverter"))
    suite.addTest(utilsTestCase("checkOneHot"))
    suite.addTest(utilsTestCase("checkAverager"))
    suite.addTest(utilsTestCase("checkAssureRatio"))
    return suite


if __name__ == "__main__":
    suite = _suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)
