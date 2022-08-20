from __future__ import annotations
from typing import cast, Sequence
import numpy as np

from redezero import types
from redezero import variable


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


def sum_to(x: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """xの要素和を求めてshapeの形状にする

    Parameters
    ----------
    x : np.ndarray
        入力配列
    shape : Sequence[int]
        出力配列の形状

    Returns
    -------
    np.ndarray
        形状変更後の配列

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = sum_to(x, (2, 1))
    >>> y
    [[ 6]
     [15]]
    """
    ndim = len(shape)
    # 入力と出力の次元差を取得
    lead = x.ndim - ndim
    # 新たに追加される軸番号を作成
    # (ブロードキャストの計算では値が小さい側に軸が追加される)
    lead_axis = tuple(range(lead))

    # もともとのshapeの軸でかつ要素が複製される軸を抽出
    # ブロードキャスト時にはlead個の軸が追加されるので、要素がふ複製される軸番号としてはi + leadとなる
    # 要素が複製される軸がない場合は空のタプルとなる
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    # 複製した要素の和を取得
    y: np.ndarray = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        # 不要な軸の削除
        y = y.squeeze(lead_axis)
    return y
