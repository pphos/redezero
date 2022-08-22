from __future__ import annotations
import numpy as np
import numpy.typing as npt

import redezero
from redezero import utils
from redezero import types
from redezero import function


class Tanh(function.Function):
    """Tanhクラス
    """
    y: npt.NDArray

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.y = utils.force_array(np.tanh(xs[0]))
        return self.y,

    def backward(self, xs: tuple[npt.NDArray, ...],
                 gys: tuple[redezero.Variable, ...]) -> tuple[redezero.Variable, ...]:
        gx = gys[0] * (1 - self.y * self.y)
        return gx


def tanh(x: types.OperandValue) -> redezero.Variable:
    """tanh関数

    Parameters
    ----------
    x : ~redezero.Variable | numpy.ndarray
        入力変数

    Returns
    -------
    ~redezero.Variable
        tanh関数を適用した結果の~redezero.Variable
    """
    return Tanh().apply((x,))[0]
