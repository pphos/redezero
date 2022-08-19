from __future__ import annotations
from typing import cast
import numpy as np

from redezero import types, variable


def force_array(x: np.ndarray | types.ScalarValue) -> np.ndarray:
    """0次元配列の演算で`numpy.ndarray`の返却を強制する

    `Function`オブジェクトが`numpy.ndarray`を返却する必要があるため, 0次元配列を変換する必要がある

    Parameters
    ----------
    x : numpy.ndarray | types.ScalarValue
        :class:`numpy.ndarray`またはスカラー値

    Returns
    -------
    numpy.ndarray
        入力値を変換した:class:`numpy.ndarray`
    """
    if np.isscalar(x):
        return np.array(x)

    return cast(np.ndarray, x)


def force_operand_value(x: types.ScalarValue | types.OperandValue) -> types.OperandValue:
    """:class:`numpy.ndarray` または :class:`~redezero.Variable`の返却を強制する

    Parameters
    ----------
    x : types.ScalarValue | types.Operand
        :class:`numpy.ndarray`, :class:`~redezero.Variable`またはスカラー値

    Returns
    -------
    types.OperandValue
        入力値を変換した:class:`numpy.ndarray`か入力値の:class:`~redezero.Variable`
    """
    converted_x = x
    if isinstance(x, variable.Variable):
        converted_x = x
    else:
        converted_x = force_array(x)

    return converted_x
