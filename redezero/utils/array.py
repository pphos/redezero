from __future__ import annotations
from typing import cast, Sequence, Optional
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


def reshape_sum_backward(gy: variable.Variable, x_shape: tuple,
                         axis: Optional[tuple[int] | int], keepdims: bool) -> variable.Variable:
    """gyの形状をx_shapeにブロードキャスト可能なように調整する

    Parameters
    ----------
    gy: Variable
        逆伝播により求めた勾配
    x_shape: tuple
        sum関数の順伝播で利用した入力の形状
    axis: Optional[tuple[int] | int]
        sum関数の順伝播で利用したaxis
    keepdims: bool
        sum関数の順伝播で利用したkeepdims

    Returns
    -------
    Variable:
        gyの形状をx_shapeに変換した勾配

    Examples
    --------
    >>> gy = Variable(np.ones(3 * 5, dtype='int64')).reshape(3, 5)
    >>> aranged_gy = reshape_sum_backward(gy, x_shape=(3, 4, 5), axis=1, keepdims=False)
    >>> aranged_gy
    [[[1 1 1 1 1],
      [1 1 1 1 1],
      [1 1 1 1 1]]]
    >>> aranged_gy.shape
    (3, 1, 5)
    """
    target_ndim = len(x_shape)

    # 和をとった後の軸番号をタプル形式にする
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    # 順伝播時に削除された軸を復元
    if not (target_ndim == 0 or tupled_axis is None or keepdims):
        # 軸を負数を使わない表記に変換
        actual_axis = [a if a >= 0 else a + target_ndim for a in cast(tuple, tupled_axis)]
        shape: list[int] = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        # 軸の調整が不要な場合は現在の形状を保持
        shape = list(gy.shape)

    gy = gy.reshape(*shape)
    return gy
