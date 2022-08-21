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

    def forward(self, x: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        y = np.tanh(x)
        return utils.force_array(y)

    def backward(self, gy: redezero.Variable) -> redezero.Variable:  # type: ignore[override]
        y_ref = self.outputs[0]

        if (y := y_ref()) is None:
            # 参照できるデータがないときはもう一度データを取得するしくみを導入する
            # 参照: Chainer: get_retained_outputs
            # https://github.com/chainer/chainer/blob/536cda7c9a146b9198f83837ba439a5afbdc074d/chainer/function_node.py#L909
            raise NotImplementedError()
        else:
            gx = gy * (1 - y * y)
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
