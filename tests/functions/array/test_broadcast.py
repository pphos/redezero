import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系broadcast_to動作():
    x = Variable(np.array([1, 2, 3]))
    expects = {
        'y_data': np.array([[1, 2, 3], [1, 2, 3]]),
        'x_grad': np.array([2, 2, 2])
    }

    # 順伝播の確認
    y = F.broadcast_to(x, (2, 3))
    assert np.array_equal(y.data, expects['y_data'])

    # 逆伝播の確認
    y.backward()
    assert np.array_equal(x.grad.data, expects['x_grad'])
