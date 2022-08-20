import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系sum_to動作():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    expects = {
        'y_data': np.array([[5, 7, 9]]),
        'x_grad': np.array([[1, 1, 1], [1, 1, 1]])
    }

    # 順伝播の確認
    y = F.sum_to(x, (1, 3))
    assert np.array_equal(y.data, expects['y_data'])

    # 逆伝播の確認
    y.backward()
    assert np.array_equal(x.grad.data, expects['x_grad'])
