import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系_sin関数動作():
    x = Variable(np.array(np.pi / 2))
    expects = {
        'x_grad': np.array(0.0),
        'y_data': np.array(1.0)
    }

    # 順伝播の確認
    y = F.sin(x)
    assert y.data == expects['y_data']

    # 逆伝播の確認
    y.backward(create_graph=True)
    assert np.isclose(x.grad.data, expects['x_grad'], atol=1e-10)


def test_正常系_cos関数動作():
    x = Variable(np.array(0.0))
    expects = {
        'x_grad': np.array(0.0),
        'y_data': np.array(1.0)
    }

    # 順伝播の確認
    y = F.cos(x)
    assert y.data == expects['y_data']

    # 逆伝播の確認
    y.backward(create_graph=True)
    assert np.isclose(x.grad.data, expects['x_grad'], atol=1e-10)
