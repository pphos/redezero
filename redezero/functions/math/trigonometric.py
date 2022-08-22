from __future__ import annotations
import numpy as np
import numpy.typing as npt

import redezero
from redezero import utils
from redezero import types
from redezero import function


class Sin(function.Function):
    """Sinクラス
    """

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        y = np.sin(xs[0])
        return utils.force_array(y),

    def backward(self, xs: tuple[npt.NDArray, ...],
                 gys: tuple[redezero.Variable, ...]) -> tuple[redezero.Variable, ...]:
        gx = gys[0] * cos(xs[0])
        return gx,


def sin(x: types.OperandValue) -> redezero.Variable:
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
    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        y = np.cos(xs[0])
        return utils.force_array(y),

    def backward(self, xs: tuple[npt.NDArray, ...],
                 gys: tuple[redezero.Variable, ...]) -> tuple[redezero.Variable, ...]:
        gx = gys[0] * -sin(xs[0])
        return gx,


def cos(x: types.OperandValue) -> redezero.Variable:
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
