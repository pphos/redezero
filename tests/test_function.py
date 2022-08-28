import pytest
import numpy as np

from redezero import function
from redezero import Variable


class FunctionApply(function.Function):
    def forward(self, xs):
        return xs


class TestFunctionApply:
    def test_apply動作_逆伝播の無効化(self):
        x = Variable(np.array(1.0))
        expect = np.array(1.0)

        with function.no_grad():
            f = FunctionApply()
            y = f.apply((x,))[0]

        assert y.data == expect
        # 逆伝播モードで使用する属性値を持っていないことを確認
        assert not hasattr(f, 'generation')
        assert not hasattr(f, 'inputs')
        assert not hasattr(f, 'outputs')

    def test_apply動作_逆伝播の有効化(self):
        x = Variable(np.array(1.0))
        expect = np.array(1.0)

        f = FunctionApply()
        y = f.apply((x,))[0]

        assert y.data == expect

        # 逆伝播モードで使用する属性値を持っていることを確認
        assert hasattr(f, 'generation')
        assert hasattr(f, 'inputs')
        assert hasattr(f, 'outputs')


class FunctionWithRetaining(function.Function):
    def forward(self, xs):
        self.retain_inputs((1,))
        self.retain_outputs((1,))
        return xs

    def backward(self, _, gys):
        self.backward_inputs = self.get_retained_inputs()
        self.backward_outputs = self.get_retained_outputs()
        return gys


class TestFunctionWithRetaining:
    @pytest.fixture
    def setup(self):
        inputs = [Variable(np.array([1])), Variable(np.array([1]))]
        self.input_data = [x.data for x in inputs]
        self.input_nodes = [x.node for x in inputs]

        self.f = FunctionWithRetaining()
        outputs = self.f.apply(inputs)
        outputs[0].grad = Variable(np.array([1]))
        outputs[0].backward()
        self.f_output_data = [y.data for y in outputs]

        del inputs

    def test_正常系_順伝播時の入力の保持(self, setup):
        assert len(self.f.backward_inputs) == 1
        assert self.f.backward_inputs[0].node == self.input_nodes[1]
        assert np.array_equal(self.f.backward_inputs[0].data, self.input_data[1])

    def test_正常系_順伝播時の出力の保持(self, setup):
        assert len(self.f.backward_outputs) == 1
        assert np.array_equal(self.f.backward_outputs[0].data, self.f_output_data[1])
