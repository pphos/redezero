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
