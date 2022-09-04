from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt

import redezero
from redezero import _backprop_utils
from redezero import types
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

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.x_shape = xs[0].shape
        y = np.broadcast_to(xs[0], self.shape)
        return y,

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)

        gx = redezero.functions.sum_to(gys[0], self.x_shape)
        return gx,


def broadcast_to(x: types.OperandValue, shape: tuple) -> redezero.Variable:
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
