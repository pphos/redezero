from __future__ import annotations
import numpy as np

import redezero
from redezero import utils
from redezero import types
from redezero import variable
from redezero import function


class SumTo(function.Function):
    """SumTo

    Attributes
    ----------
    shape : tuple
        出力したい配列の形状
    x_shape: tuple
        順伝播時の配列の形状
    """
    shape: tuple
    x_shape: tuple

    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: variable.Variable) -> variable.Variable:  # type: ignore[override]
        gx = redezero.functions.broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x: types.OperandValue, shape: tuple) -> variable.Variable:
    """軸に沿って配列の要素を足し合わせ, 指定された形状の配列を出力する

    Parameters
    ----------
    x : numpy.ndarray | ~redezero.Variable
        入力変数
    shape : tuple
        出力したい配列の形状

    Returns
    -------
    ~redezero.Variable
        `shape`の形状になるように要素を足し合わせたVariableインスタンス

    Examples
    --------
    >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> x
    array([[1., 2., 3.],
           [4., 5., 6.]])
    >>> y = sum_to(x, (1, 3))
    >>> y
    variable([[5., 7., 9.]])
    >>> z = sum_to(x, (2, 1))
    >>> z
    variable([[ 6.],
              [15.]])
    """
    if x.shape == shape:
        return redezero.as_variable(x)
    return SumTo(shape).apply((x,))[0]
