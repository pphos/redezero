import pytest
import numpy as np

from redezero import utils
from redezero import Variable


def params_for_force_array():
    """正常系_force_arrayのパラメータ
    """
    return {
        '引数がnumpy.ndarray': (np.array(1)),
        '引数がint': (1),
        '引数がfloat': (1.0)
    }


@pytest.mark.parametrize(
    'x',
    params_for_force_array().values(),
    ids=params_for_force_array().keys()
)
def test_正常系_force_array(x):
    converted_x = utils.force_array(x)
    assert isinstance(converted_x, np.ndarray)


def params_for_force_operand_value():
    """正常系_force_operand_valueのパラメータ
    """
    return {
        '引数がVariable': (Variable(np.array(1.0))),
        '引数がnumpy.ndarray': (np.array(1.0)),
        '引数がint': (1),
        '引数がfloat': (1.0)
    }


@pytest.mark.parametrize(
    'x',
    params_for_force_operand_value().values(),
    ids=params_for_force_operand_value().keys()
)
def test_正常系_force_operand_value(x):
    converted_x = utils.force_operand_value(x)
    assert isinstance(converted_x, Variable) | isinstance(converted_x, np.ndarray)


def params_for_sum_to():
    """sum_to動作のパラメータ
    """
    x = np.ones(3 * 4 * 5, dtype='int64').reshape(3, 4, 5)

    return {
        '全ての要素の足し合わせ': (
            x,
            (),
            np.array(60)
        ),
        '0軸方向に足し合わせ': (
            x,
            (1, 4, 5),
            np.array([
                [[3, 3, 3, 3, 3],
                 [3, 3, 3, 3, 3],
                 [3, 3, 3, 3, 3],
                 [3, 3, 3, 3, 3]
                 ]]
            )
        ),
        '0軸と1軸方向に足し合わせ': (
            x,
            (1, 1, 5),
            np.array([[[12, 12, 12, 12, 12]]])
        )
    }


@pytest.mark.parametrize(
    'x, shape, expect',
    params_for_sum_to().values(),
    ids=params_for_sum_to().keys()
)
def test_sum_to動作(x, shape, expect):
    x = np.ones(3 * 4 * 5, dtype='int64').reshape(3, 4, 5)
    y = utils.sum_to(x, shape)
    assert np.array_equal(y, expect)


def params_for_reshape_sum_backward():
    """正常系_reshape_sum_backward動作のパラメータ
    """
    x_shape = (3, 4, 5)
    gy = Variable(np.ones(3 * 5, dtype='int64').reshape(3, 5))

    return {
        '正常系_軸の調整が必要': (
            gy, x_shape, 1, False, gy.reshape(3, 1, 5).data
        ),
        '正常系_軸の調整が不要': (
            gy, x_shape, 1, True, gy.data
        )
    }


@pytest.mark.parametrize(
    'gy, x_shape, axis, keepdims, expect_aranged_gy_data',
    params_for_reshape_sum_backward().values(),
    ids=params_for_reshape_sum_backward().keys()
)
def test_正常系_reshape_sum_backward動作(gy, x_shape, axis, keepdims, expect_aranged_gy_data):
    # 逆伝播の入力の形状を調整
    aranged_gy = utils.reshape_sum_backward(gy, x_shape, axis, keepdims)

    assert np.array_equal(aranged_gy.data, expect_aranged_gy_data)
