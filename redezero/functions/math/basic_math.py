from __future__ import annotations
import numpy as np
import numpy.typing as npt

import redezero
from redezero import types
from redezero import function
from redezero import utils


class Add(function.Function):
    """加算クラス

    Attributes
    ----------
    x0_shape: tuple
        左項の形状
    x1_shape: tuple
        右項の形状
    """
    x0_shape: tuple
    x1_shape: tuple

    def forward(self, x0: npt.NDArray, x1: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return utils.force_array(y)

    def backward(self, gy: redezero.Variable) -> tuple[redezero.Variable, ...]:  # type: ignore[override]
        gx0, gx1 = gy, gy
        # 形状が異なる場合にはブロードキャスト
        if self.x0_shape != self.x1_shape:
            gx0 = redezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = redezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0: types.OperandValue, x1: types.ScalarValue | types.OperandValue) -> redezero.Variable:
    """加算

    Parameters
    ----------
    x0 : numpy.ndarray | ~redezero.Variable
        演算の左項
    x1 : types.ScalarValue | numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        加算の結果の~redezero.Variable
    """

    converted_x1 = utils.force_operand_value(x1)
    return Add().apply((x0, converted_x1))[0]


class Mul(function.Function):
    """乗算クラス
    """

    def forward(self, x0: npt.NDArray, x1: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        y = x0 * x1
        return utils.force_array(y)

    def backward(self, gy: redezero.Variable) -> tuple[redezero.Variable, ...]:  # type: ignore[override]
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        # 形状が異なる場合にはブロードキャスト
        if x0.shape != x1.shape:
            gx0 = redezero.functions.sum_to(gx0, x0.shape)
            gx1 = redezero.functions.sum_to(gx1, x1.shape)

        return gx0, gx1


def mul(x0: types.OperandValue, x1: types.ScalarValue | types.OperandValue) -> redezero.Variable:
    """乗算

    Parameters
    ----------
    x0 : numpy.ndarray | ~redezero.Variable
        演算の左項
    x1 : types.ScalarValue | numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        乗算の結果の~redezero.Variable
    """
    converted_x1 = utils.force_operand_value(x1)
    return Mul().apply((x0, converted_x1))[0]


class Neg(function.Function):
    """負数クラス
    """

    def forward(self, x: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        return utils.force_array(-x)

    def backward(self, gy: redezero.Variable) -> tuple[redezero.Variable, ...]:  # type: ignore[override]
        return -gy,


def neg(x: types.OperandValue) -> redezero.Variable:
    """負数

    Parameters
    ----------
    x : numpy.ndarray | ~redezero.Variable
        演算の項

    Returns
    -------
    ~redezero.Variable
        負数の結果の~redezero.Variable
    """
    return Neg().apply((x,))[0]


class Sub(function.Function):
    """減算クラス

    Attributes
    ----------
    x0_shape: tuple
        左項の形状
    x1_shape: tuple
        右項の形状
    """
    x0_shape: tuple
    x1_shape: tuple

    def forward(self, x0: npt.NDArray, x1: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return utils.force_array(y)

    def backward(self, gy: redezero.Variable) -> tuple[redezero.Variable, ...]:  # type: ignore[override]
        gx0 = gy
        gx1 = -gy
        # 形状が異なる場合にはブロードキャスト
        if self.x0_shape != self.x1_shape:
            gx0 = redezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = redezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0: types.OperandValue, x1: types.ScalarValue | types.OperandValue) -> redezero.Variable:
    """減算

    Parameters
    ----------
    x0 : numpy.ndarray | ~redezero.Variable
        演算の左項
    x1 : types.ScalarValue | numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        減算の結果の~redezero.Variable
    """
    converted_x1 = utils.force_operand_value(x1)
    return Sub().apply((x0, converted_x1))[0]


def rsub(x0: types.OperandValue, x1: types.ScalarValue | types.OperandValue) -> redezero.Variable:
    """減算(右辺)

    Parameters
    ----------
    x0 : numpy.ndarray | ~redezero.Variable
        演算の左項
    x1 : types.ScalarValue | numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        減算の結果の~redezero.Variable
    """
    converted_x1 = utils.force_operand_value(x1)
    return Sub().apply((converted_x1, x0))[0]


class Div(function.Function):
    """除算クラス
    """

    def forward(self, x0: npt.NDArray, x1: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        y = x0 / x1
        return utils.force_array(y)

    def backward(self, gy: redezero.Variable) -> tuple[redezero.Variable, ...]:  # type: ignore[override]
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        # 形状が異なる場合にはブロードキャスト
        if x0.shape != x1.shape:
            gx0 = redezero.functions.sum_to(gx0, x0.shape)
            gx1 = redezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0: types.OperandValue, x1: types.ScalarValue | types.OperandValue) -> redezero.Variable:
    """除算

    Parameters
    ----------
    x0 : numpy.ndarray | ~redezero.Variable
        演算の左項
    x1 : types.ScalarValue | numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        除算の結果の~redezero.Variable
    """
    converted_x1 = utils.force_operand_value(x1)
    return Div().apply((x0, converted_x1))[0]


def rdiv(x0: types.OperandValue, x1: types.ScalarValue | types.OperandValue) -> redezero.Variable:
    """除算 (右辺)

    Parameters
    ----------
    x0 : numpy.ndarray | ~redezero.Variable
        演算の左項
    x1 : types.ScalarValue | numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        除算の結果の~redezero.Variable
    """
    converted_x1 = utils.force_operand_value(x1)
    return Div().apply((converted_x1, x0))[0]


class Pow(function.Function):
    """累乗クラス
    """

    def __init__(self, c: int) -> None:
        self.c: int = c

    def forward(self, x: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        y = x ** self.c
        return utils.force_array(y)

    def backward(self, gy: redezero.Variable) -> redezero.Variable:  # type: ignore[override]
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: types.OperandValue, c: int) -> redezero.Variable:
    """累乗

    Parameters
    ----------
    x : numpy.ndarray | ~redezero.Variable
        底となる項
    c : int
        乗数

    Returns
    -------
    ~redezero.Variable
        累乗の結果の~redezero.Variable
    """
    return Pow(c).apply((x,))[0]
