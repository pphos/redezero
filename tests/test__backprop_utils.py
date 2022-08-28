import pytest
import numpy as np

from redezero import Variable, _backprop_utils


def params_for_reduce():
    """正常系_勾配集計のパラメータ"""

    return {
        'リストが空': ([], None),
        'リストに1つ勾配が存在': (
            [
                Variable(np.array([1]))
            ],
            np.array([1])
        ),
        'リストに2つ勾配が存在': (
            [
                Variable(np.array([1])),
                Variable(np.array([1]))
            ],
            np.array([2])
        ),
    }


@pytest.mark.parametrize(
    'grads, expect',
    params_for_reduce().values(),
    ids=params_for_reduce().keys()
)
def test_正常系_勾配集計(grads, expect):
    reduced_grad = _backprop_utils._reduce(grads)

    if expect is None:
        assert reduced_grad is None
    else:
        assert np.array_equal(reduced_grad.data, expect)
