from __future__ import annotations
import numpy as np

import redezero
from redezero import types
from redezero import variable
from redezero import function


class BroadCastTo(function.Function):
    """ブロードキャスト

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
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: variable.Variable) -> variable.Variable:  # type: ignore[override]
        gx = redezero.functions.sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: types.OperandValue, shape: tuple) -> variable.Variable:
    """入力変数をshapeの形状に沿ってブロードキャストする

    Parameters
    ----------
    x : numpy.ndarray | ~redezero.Variable
        入力変数
    shape : tuple
        出力したい配列の形状

    Returns
    -------
    ~redezero.Variable
        `shape`の形状になるようにブロードキャストしたVariableインスタンス

    Examples
    --------
    >>> x = np.arange(0, 3)
    >>> x
    array([0, 1, 2])
    >>> y = F.broadcast_to(x, (3, 3))
    >>> y.array
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]])
    """
    if x.shape == shape:
        return redezero.as_variable(x)
    return BroadCastTo(shape).apply((x,))[0]
