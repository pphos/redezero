from __future__ import annotations
from typing import Optional
import numpy.typing as npt

import redezero
from redezero import _backprop_utils
from redezero import utils
from redezero import types
from redezero import function


class MatMul(function.Function):
    """MatMal
    """

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.retain_inputs((0, 1))
        x, W = xs
        y = x.dot(W)
        return utils.force_array(y),

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)
        x, W = self.get_retained_inputs()
        gx = matmul(gys[0], W.T)
        gW = matmul(x.T, gys[0])
        return gx, gW


def matmul(x: types.OperandValue, W: types.OperandValue) -> redezero.Variable:
    """行列の積

    Parameters
    ----------
    x : numpy.ndarray | ~redezero.Variable
        演算の左項
    W : numpy.ndarray | ~redezero.Variable
        演算の右項

    Returns
    -------
    ~redezero.Variable
        行列の積の演算結果
    """
    return MatMul().apply((x, W))[0]
