from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence

import redezero
from redezero import function
from redezero import types


class Transpose(function.Function):
    """行列の転置

    Attributes
    ----------
    axes : Optional[Sequence[int]]
        転置後の軸の順番
    """
    axes: Optional[Sequence[int]]

    def __init__(self, axes: Optional[Sequence[int]] = None):
        self.axes = axes

    def forward(self, x: npt.NDArray) -> npt.NDArray:  # type: ignore[override]
        y = x.transpose(self.axes)
        return y

    def backward(self, gy: redezero.Variable) -> redezero.Variable:  # type: ignore[override]
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        # 逆伝播では順伝播と逆向きに軸を置き換える
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x: types.OperandValue, axes: Optional[Sequence[int]] = None) -> redezero.Variable:
    """行列の転置を実施

    Parameters
    ----------
    x: : numpy.ndarray | ~redezero.Variable
        転置対象の入力変数
    axes : Optional[Sequence[int]]
        転置を行なった後のの入れ替え方を指定する
        Noneの場合には軸の順番を逆にする

    Returns
    ----------
    ~redezero.Variable
        入力変数を転置したVariable

    Examples
    -----------
    >>> x = np.array([[[0, 1, 2], [3, 4, 5]]])
    >>> x.shape
    (1, 2, 3)
    >>> y = F.transpose(x)
    >>> y.shape
    (3, 2, 1)
    >>> y = transpose(x, axes=(1, 0, 2))
    >>> y.shape
    (2, 1, 3)
    """
    return Transpose(axes).apply((x,))[0]
