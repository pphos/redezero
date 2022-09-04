from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt

import redezero
from redezero import _backprop_utils
from redezero import utils
from redezero import types
from redezero import function


class Sin(function.Function):
    """Sinクラス
    """

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.retain_inputs((0,))

        y = np.sin(xs[0])
        return utils.force_array(y),

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)

        x = self.get_retained_inputs()[0]
        gx = gys[0] * cos(x)
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
        self.retain_inputs((0,))

        y = np.cos(xs[0])
        return utils.force_array(y),

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)

        x = self.get_retained_inputs()[0]
        gx = gys[0] * -sin(x)
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
