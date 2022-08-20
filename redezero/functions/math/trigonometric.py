from __future__ import annotations
import numpy as np

from redezero import utils
from redezero import types
from redezero import variable
from redezero import function


class Sin(function.Function):
    """Sinクラス
    """

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.sin(x)
        return utils.force_array(y)

    def backward(self, gy: variable.Variable) -> variable.Variable:  # type: ignore[override]
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x: types.OperandValue) -> variable.Variable:
    """sin関数

    Parameters
    ----------
    x : ~redezero.Variable | numpy.ndarray
        入力変数

    Returns
    -------
    ~redezero.Variable
        sin関数を適用した結果の~redezero.Variable
    """
    return Sin().apply((x,))[0]


class Cos(function.Function):
    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.cos(x)
        return utils.force_array(y)

    def backward(self, gy: variable.Variable) -> variable.Variable:  # type: ignore[override]
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x: types.OperandValue) -> variable.Variable:
    """cos関数

    Parameters
    ----------
    x : ~redezero.Variable | numpy.ndarray
        入力変数

    Returns
    -------
    ~redezero.Variable
        cos関数を適用した結果の~redezero.Variable
    """
    return Cos().apply((x,))[0]
