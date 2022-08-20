from __future__ import annotations
from typing import cast
import numpy as np

from redezero import utils
from redezero import types
from redezero import variable
from redezero import function


class Tanh(function.Function):
    """Tanhクラス
    """

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        y = np.tanh(x)
        return utils.force_array(y)

    def backward(self, gy: variable.Variable) -> variable.Variable:  # type: ignore[override]
        casted_y = cast(variable.Variable, self.outputs[0]())
        gx = gy * (1 - casted_y * casted_y)
        return gx


def tanh(x: types.OperandValue) -> variable.Variable:
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
