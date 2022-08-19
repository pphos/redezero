import pytest
import numpy as np

from redezero import Variable


# 足し算
def params_for_add():
    """正常系_加算_項が全てVariableのパラメータ
    """
    return {
        'ブロードキャストなし': (
            Variable(np.array(2.0)),
            Variable(np.array(3.0)),
            {
                'x0_grad': np.array(1.0),
                'x1_grad': np.array(1.0),
                'y_data': np.array(5.0)
            }
        ),
    }


@pytest.mark.parametrize(
    'x0, x1, expects',
    params_for_add().values(),
    ids=params_for_add().keys()
)
def test_正常系_加算_項が全てVariable(x0, x1, expects):
    # 順伝播の確認
    y = x0 + x1
    assert np.array_equal(y.data, expects['y_data'])

    # 逆伝播の確認
    y.backward()
    assert np.array_equal(x0.grad.data, expects['x0_grad'])
    assert np.array_equal(x1.grad.data, expects['x1_grad'])


def params_for_add_left_operand_is_not_variable():
    """正常系_加算_左項がVariableでないのパラメータ
    """
    return {
        '左項がndarray': (
            np.array(2.0),
            Variable(np.array(3.0)),
            np.array(5.0)
        ),
        '左項がint': (
            2,
            Variable(np.array(3.0)),
            np.array(5.0)
        ),
        '左項がfloat': (
            2.0,
            Variable(np.array(3.0)),
            np.array(5.0)
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expect_y',
    params_for_add_left_operand_is_not_variable().values(),
    ids=params_for_add_left_operand_is_not_variable().keys()
)
def test_正常系_加算_左項がVariableでない(x0, x1, expect_y):
    # 順伝播の確認
    y = x0 + x1
    assert y.data == expect_y


# 掛け算
def params_for_mul():
    """正常系_乗算_項が全てVariableのパラメータ
    """
    return {
        'ブロードキャストなし': (
            Variable(np.array(2.0)),
            Variable(np.array(3.0)),
            {
                'x0_grad': np.array(3.0),
                'x1_grad': np.array(2.0),
                'y_data': np.array(6.0)
            }
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expects',
    params_for_mul().values(),
    ids=params_for_mul().keys()
)
def test_正常系_乗算_項が全てVariable(x0, x1, expects):
    # 順伝播の確認
    y = x0 * x1
    assert np.array_equal(y.data, expects['y_data'])

    # 逆伝播の確認
    y.backward()
    assert np.array_equal(x0.grad.data, expects['x0_grad'])


def params_for_mul_left_term_is_not_variable():
    """正常系_乗算_左項がVariableでないのパラメータ
    """
    return {
        '正常系_左項がndarray': (
            np.array(3.0),
            Variable(np.array(2.0)),
            np.array(6.0)
        ),
        '正常系_左項がint': (
            3,
            Variable(np.array(2.0)),
            np.array(6.0)
        ),
        '正常系_左項がfloat': (
            3.0,
            Variable(np.array(2.0)),
            np.array(6.0)
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expect_y',
    params_for_mul_left_term_is_not_variable().values(),
    ids=params_for_mul_left_term_is_not_variable().keys()
)
def test_正常系_乗算_左項がVariableでない(x0, x1, expect_y):
    # 順伝播の確認
    y = x0 * x1
    assert y.data == expect_y


def test_正常系_負数():
    x = Variable(np.array(2.0))
    expects = {
        'x_grad': np.array(-1.0),
        'y_data': np.array(-2.0)
    }

    # 順伝播の確認
    y = -x
    assert y.data == expects['y_data']

    # 逆伝播の確認
    y.backward()
    assert x.grad.data == expects['x_grad']


# 引き算
def params_for_sub():
    """正常系_減算_項が全てVariableのパラメータ
    """
    return {
        'ブロードキャストなし': (
            Variable(np.array(2.0)),
            Variable(np.array(1.0)),
            {
                'x0_grad': np.array(1.0),
                'x1_grad': np.array(-1.0),
                'y_data': np.array(1.0)
            }
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expects',
    params_for_sub().values(),
    ids=params_for_sub().keys()
)
def test_正常系_減算_項が全てVariable(x0, x1, expects):
    # 順伝播の確認
    y = x0 - x1
    assert y.data == expects['y_data']

    # 逆伝播の確認
    y.backward()
    assert x0.grad.data == expects['x0_grad']
    assert x1.grad.data == expects['x1_grad']


def params_for_sub_left_term_is_not_variable():
    """正常系_減算_左項がVariableでない
    """

    return {
        '正常系_左項がndarray': (
            np.array(2.0),
            Variable(np.array(1.0)),
            np.array(1.0)
        ),
        '正常系_左項がint': (
            2,
            Variable(np.array(1.0)),
            np.array(1.0)
        ),
        '正常系_左項がfloat': (
            2.0,
            Variable(np.array(1.0)),
            np.array(1.0)
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expect_y',
    params_for_sub_left_term_is_not_variable().values(),
    ids=params_for_sub_left_term_is_not_variable().keys()
)
def test_正常系_減算_左項がVariableでない(x0, x1, expect_y):
    # 順伝播の確認
    y = x0 - x1
    assert y.data == expect_y


# 割り算
def params_for_div():
    """正常系_除算_項が全てVariableのパラメータ
    """
    return {
        'ブロードキャストなし': (
            Variable(np.array(4.0)),
            Variable(np.array(2.0)),
            {
                'x0_grad': np.array(0.5),
                'x1_grad': np.array(-1.0),
                'y_data': np.array(2.0)
            }
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expects',
    params_for_div().values(),
    ids=params_for_div().keys()
)
def test_正常系_除算_項が全て_Variable(x0, x1, expects):
    # 順伝播の確認
    y = x0 / x1
    assert y.data == expects['y_data']

    # 逆伝播の確認
    y.backward()
    assert x0.grad.data == expects['x0_grad']
    assert x1.grad.data == expects['x1_grad']


def params_for_div_left_operand_is_not_variable():
    """正常系_左項がVariableでないのパラメータ
    """
    return {
        '左項がndarray': (
            np.array(4.0),
            Variable(np.array(2.0)),
            np.array(2.0)
        ),
        '左項がint': (
            4,
            Variable(np.array(2.0)),
            np.array(2.0)
        ),
        '左項がfloat': (
            4.0,
            Variable(np.array(2.0)),
            np.array(2.0)
        )
    }


@pytest.mark.parametrize(
    'x0, x1, expect_y',
    params_for_div_left_operand_is_not_variable().values(),
    ids=params_for_div_left_operand_is_not_variable().keys()
)
def test_正常系_除算_左項がVariableでない(x0, x1, expect_y):
    """正常系_除算_左項がVariableでない
    """
    # 順伝播の確認
    y = x0 / x1
    assert y.data == expect_y


# 累乗
def test_正常系_累乗():
    x = Variable(np.array(2.0))
    y = x ** 3
    expects = {
        'x_grad': np.array(12.0),
        'y_data': np.array(8.0)
    }

    # 順伝播の確認
    y = x ** 3
    assert y.data == expects['y_data']

    # 逆伝播の確認
    y.backward()
    assert x.grad.data == expects['x_grad']
