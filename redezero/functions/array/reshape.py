from __future__ import annotations
import numpy as np

from redezero import variable
from redezero import function
from redezero import types


class Reshape(function.Function):
    """入力配列の形状変更をコピーなしで実施

    Attributes
    ----------
    x_shape: tuple
        入力配列の元の形状
    shape: tuple
        変換したい配列の形状
    """
    x_shape: tuple
    shape: tuple

    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: variable.Variable) -> variable.Variable:  # type: ignore[override]
        return reshape(gy, self.x_shape)


def reshape(x: types.OperandValue, shape: tuple) -> variable.Variable:
    """入力配列の形状変更をコピーなしで実施

    Parameters
    ----------
    x : numpy.ndarray | ~redezero.Variable
        形状変更した入力変数
    shape : tuple
        期待する出力変数の形状

    Returns
    -------
    ~redezero.Variable
        入力変数をshapeの形状に変換したVariable

    Examples
    --------
    >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> y = reshape(x, (8,))
    >>> y.shape
    (8,)
    >>> y.array
    array([1, 2, 3, 4, 5, 6, 7, 8])
    """
    if x.shape == shape:
        return variable.as_variable(x)
    return Reshape(shape).apply((x,))[0]
