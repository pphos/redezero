from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt

import redezero
from redezero import _backprop_utils
from redezero import utils
from redezero import types
from redezero import function


class Tanh(function.Function):
    """Tanhクラス
    """

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        self.retain_outputs((0,))
        y = utils.force_array(np.tanh(xs[0]))
        return y,

    def backward(self, _, gys: tuple[Optional[redezero.Variable], ...]) -> tuple[Optional[redezero.Variable], ...]:
        gys = _backprop_utils.preprocess_backward_grad_outputs(gys)

        y = self.get_retained_outputs()[0]

        gx = gys[0] * (1 - y * y)
        return gx,


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
