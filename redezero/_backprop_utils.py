from __future__ import annotations
from typing import Optional
import functools

import redezero
from redezero import function


def _reduce(grads: list[redezero.Variable]) -> Optional[redezero.Variable]:
    """リスト内の勾配を集計する

    ``grads``が[]の場合, ``None``を返却する

    Parameters
    ----------
    grads : list[~redezero.Variable]
        勾配のリスト

    Returns
    -------
    Optional[~redezero.Variable]
        集計後の勾配
        ``grads``が``[]``の場合には``None``が返却される
    """
    if not grads:
        return None

    reduced_grad = grads[0]
    if len(grads) >= 2:
        reduced_grad = functools.reduce(lambda x, y: x + y, grads)

    return reduced_grad


def _pure(grad: Optional[redezero.Variable]) -> list[redezero.Variable]:
    """勾配をリストにしたものを返却する

    ``grad``が``None``の場合には空リストを返却する

    Parameters
    ----------
    grad : Optional[~redezero.Variable]
        勾配

    Returns
    -------
    list[~redezero.Variable]
        リストに変換した勾配
        勾配が``None``に場合には``[]``が返却される
    """
    return [] if grad is None else [grad]


def _pop_or_none(grads: list[redezero.Variable]) -> Optional[redezero.Variable]:
    """リストの末尾の勾配を取得する

    ``grads``が``[]``の場合には``None``を返却する

    Parameters
    ----------
    grads : list[~redezero.Variable]
        勾配のリスト

    Returns
    -------
    Optional[redezero.Variable]
        リストの末尾から取得された勾配.
        空リストの場合にはNoneが返却される
    """
    return grads.pop() if grads else None


class GradTable:
    """勾配を参照するためのノードを保持するクラス

    参照: Chainer _backprop_utils.py
    https://github.com/chainer/chainer/blob/v5/chainer/_backprop_utils.py

    勾配は逆伝播の過程で参照として保存される.
    勾配を厳密に累積するためには保持するリスト長を1以下に保つ必要がある

    Attributes
    ----------
    grads : dict[redezero.VariableNode, list[redezero.Variable]]
        VariableNodeに対応する勾配を保持した辞書
    """
    grads: dict[redezero.VariableNode, list[redezero.Variable]]
    _load_if_new: bool

    def __init__(self, load_if_new: bool = False) -> None:
        """インスタンスの初期化

        Parameters
        ----------
        load_if_new : bool
            ノードが追加されていない場合, ノードの``grad``を読み取るかのフラグ
        """
        self.grads = {}
        self._load_if_new = load_if_new

    def __setitem__(self, node: redezero.VariableNode, grad: Optional[redezero.Variable]) -> None:
        """勾配辞書への勾配の設定

        ``grad``が``None``の場合には辞書の値として``[]``が設定される

        Parameters
        ----------
        node : ~redezero.VariableNode
            辞書のkeyとなるVariableNode
        grad : Optional[~redezero.Variable]
            辞書のvalueとなる勾配
        """
        self.grads[node] = _pure(grad)

    def get_as_list(self, node: redezero.VariableNode) -> list[redezero.Variable]:
        """``node``に対応する勾配リストを取得

        Parameters
        ----------
        node : redezero.VariableNode
            勾配を取得したいVariableNode

        Returns
        -------
        list[redezero.VariableNode]
            VariableNodeに対応する勾配のリスト
        """
        grads = self.grads
        if node not in grads:
            if self._load_if_new and node.creator is None:
                # nodeが勾配の末端である場合にのみ勾配を累積する
                grads[node] = _pure(node.grad)
            else:
                grads[node] = []
        return grads[node]

    def pop(self, node: Optional[redezero.VariableNode]) -> Optional[redezero.Variable]:
        """勾配スタックから勾配を取得する

        Parameters
        ----------
        node : Optional[~redezero.VariableNode]
            勾配を取得したいVariableNode

        Returns
        -------
        Optional[redezero.Varielse:
            return None]
            ``node``に対応する勾配
            ``node``が``None``の場合には``None``が返却される
        """
        if node is None:
            return None

        grads = self.grads
        if node in grads:
            # 集計した勾配を返却する
            return _reduce(grads.pop(node))

        if self._load_if_new:
            return node.grad
        else:
            return None

    def assert_no_grad(self) -> None:
        """勾配辞書に勾配が存在しないことを確認"""
        for gx in self.grads.values():
            assert gx == []


def backprop_step(func: function.Function, target_input_indexes: tuple[int, ...],
                  grad_outputs: tuple[redezero.Variable, ...],
                  grad_inputs: dict[redezero.VariableNode, list[redezero.Variable]]) -> None:
    """Functionの勾配を累積する

    この処理は:meth:`~redezero.Variable.backward`で利用される.
    逆伝播の実処理は:class:`~redezero.Function`クラスの:meth:`backward_accumulate`を参照

    Parameters
    ----------
    func : ~redezero.Function
        勾配が累積されるFunction
    target_input_indexes : tuple[int, ...]
        ソート済みの勾配累積が必要な入力のインデックスのタプル
    grad_outputs : tuple[~redezero.Variable, ...]
        出力変数に関する勾配
        出力変数に関する勾配が与えられない場合, 対応する要素は``None``である
    grad_inputs : dict[~redezero.VariableNode, list[~redezero.Variable]]
        入力変数に関する勾配への参照
    """
    grad_inputs_tuple = tuple([
        _pop_or_none(grad_inputs[func.inputs[i]])
        for i in target_input_indexes
    ])
    gxs = func.backward_accumulate(target_input_indexes, grad_outputs, grad_inputs_tuple)

    len_gxs = len(gxs)
    if len_gxs == len(func.inputs):
        gxs = tuple([gxs[i] for i in target_input_indexes])
    elif len_gxs != len(target_input_indexes):
        msg = 'number of gradients returned from backward is incorrect: '
        if len(func.inputs) == len(target_input_indexes):
            msg += f"{len_gxs} != expected {len(func.inputs)}"
        else:
            msg += f"{len_gxs} != expected {len(func.inputs)} or {len(target_input_indexes)}"
        raise ValueError(msg)

    # 逆伝播で求めた勾配を入力変数の勾配辞書に追加
    for i, gx in zip(target_input_indexes, gxs):
        if gx is not None:
            grad_inputs[func.inputs[i]].append(gx)
    del gxs

    # 勾配の累積
    for gx in grad_inputs.values():
        _reduce(gx)
