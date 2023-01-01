import pytest
import numpy as np

from redezero import Variable
from redezero import functions as F


def params_for_sum():
    """正常系_sum動作
    """
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    axis = (0,)
    return {
        '入力と出力が同じ次元': (
            x,
            axis,
            True,
            {
                'y_data': np.array([[5, 7, 9]]),
                'x_grad': np.array([[1, 1, 1], [1, 1, 1]])
            }
        ),
        '入力と出力の違う次元': (
            x,
            axis,
            False,
            {
                'y_data': np.array([5, 7, 9]),
                'x_grad': np.array([[1, 1, 1], [1, 1, 1]])
            }
        )
    }


@pytest.mark.parametrize(
    'x, axis, keepdims, expects',
    params_for_sum().values(),
    ids=params_for_sum().keys()
)
def test_正常系sum動作(x, axis, keepdims, expects):
    # 順伝播の確認
    y = F.sum(x, axis, keepdims=keepdims)
    assert np.array_equal(y.data, expects['y_data'])

    # 逆伝播の確認
    y.backward()
    assert np.array_equal(x.grad.data, expects['x_grad'])


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
