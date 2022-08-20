import pytest
import numpy as np

from redezero import utils
from redezero import function
from redezero import Variable
from redezero import configuration


class SimpleFunction(function.Function):
    def forward(self, x):
        return x

    def backward(self, gy):
        pass


class TestFunction:
    def test_apply動作_逆伝播の無効化(self):
        x = Variable(np.array(1.0))
        expect = np.array(1.0)

        with configuration.no_grad():
            f = SimpleFunction()
            y = f.apply((x,))[0]

        assert y.data == expect
        # 逆伝播モードで使用する属性値を持っていないことを確認
        assert not hasattr(f, 'generation')
        assert not hasattr(f, 'inputs')
        assert not hasattr(f, 'outputs')

    def test_apply動作_逆伝播の有効化(self):
        x = Variable(np.array(1.0))
        expect = np.array(1.0)

        f = SimpleFunction()
        y = f.apply((x,))[0]

        assert y.data == expect

        # 逆伝播モードで使用する属性値を持っていることを確認
        assert hasattr(f, 'generation')
        assert hasattr(f, 'inputs')
        assert hasattr(f, 'outputs')
