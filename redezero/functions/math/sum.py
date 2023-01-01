from __future__ import annotations
from typing import Optional
import numpy.typing as npt

import redezero
from redezero import _backprop_utils
from redezero import utils
from redezero import types
from redezero import function
from redezero.functions.array.broadcast import broadcast_to


class Sum(function.Function):
    """Sum

    Attributes
    ----------
    axis : tuple
        和を求める際の軸
    keepdims : bool
        入力と出力を同じ次元数にするかのフラグ
    x_shape: tuple
        順伝播時の配列の形状
    """
    axis: Optional[tuple]
    keepdims: bool

    def __init__(self, axis: Optional[tuple], keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.x_shape = xs[0].shape
        y = xs[0].sum(axis=self.axis, keepdims=self.keepdims)
        return utils.force_array(y),

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)
        gy = utils.reshape_sum_backward(gys[0], self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)

        return gx,


def sum(x: types.OperandValue, axis=Optional[tuple], keepdims: bool = False) -> redezero.Variable:
    """axisの軸に沿って要素を足し合わせる
    keepdimsを指定した場合には, 入力と出力の次元を同じにする

    Parameters
    ----------
     x : numpy.ndarray | ~redezero.Variable
        入力変数
    axis : Optional[tuple]
        和を求める際の軸
    keepdims : bool
        入力と出力を同じ次元数にするかのフラグ

    Returns
    ----------
    ~redezero.Variable
        axis軸に沿って要素を足し合わせた後のVariableインスタンス

    Examples
    ---------
    >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    >>> y = sum(x, axis=(0,))
    >>> y
    variables([5, 7, 9])
    """
    return Sum(axis, keepdims).apply((x,))[0]


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

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.x_shape = xs[0].shape
        y = utils.sum_to(xs[0], self.shape)
        return y,

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)

        gx = redezero.functions.broadcast_to(gys[0], self.x_shape)
        return gx,


def sum_to(x: types.OperandValue, shape: tuple) -> redezero.Variable:
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
