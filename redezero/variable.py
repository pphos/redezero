from __future__ import annotations
from weakref import ReferenceType, ref
from typing import Optional, Sequence
import heapq
import numpy as np
import numpy.typing as npt

import redezero
from redezero import _backprop_utils
from redezero import configuration
from redezero import function
from redezero.functions.math import basic_math


class VariableNode:
    """Variableの逆伝播の計算グラフを表すノード

    参照: Chainer variable.py
    https://github.com/chainer/chainer/blob/238b1c0978f506a8c35b685ebd927b9885b648b3/chainer/variable.py

    このオブジェクトはvariable nodeの計算グラフを表し,
    誤差逆伝播で使用される各々の関数に渡される勾配を決定する

    variable nodeは:class:`Variable`オブジェクトにより保持される.
    :class:`Function`オブジェクトはvariableを引数として取り, variable nodeへの参照も保持する.
    注意として, 一般にvariable nodeは対応するdata配列への参照を保持しない.

    実際には, 以下のケースではdata配列はvariable nodeからアクセスできます.
    1.  variable nodeへの参照を保持する:class:`Variable`オブジェクトが存在する場合,
        variable nodeはvariableオブジェクトへの弱参照を保持するため, 弱参照を介してdata配列にアクセスできる.
    2.  :meth:`retain_data`が呼び出された場合, variable nodeはdata配列への参照を保持する.
        主に入力または出力のdata配列を必要とする誤差逆伝播の手順中で呼び出される.

    Attributes
    ----------
    dtype : numpy.dtype
        data配列の型
    shape : tuple
        data配列の形状
    name : Optional[str]
        variable nodeの名前
    """
    _variable: ReferenceType[Variable]
    _creator: Optional[function.Function]
    _data: Optional[npt.NDArray]
    _generation: int
    dtype: np.dtype
    ndim: int
    shape: tuple
    size: int
    name: Optional[str]

    def __init__(self, variable: Variable, name: Optional[str]) -> None:
        """VariableNodeインスタンスの初期化

        Parameters
        ----------
        variable : ~redezero.Variable
            入力のVariableオブジェクト
        name : Optional[str]
            variable nodeの名前
        """
        self._variable = ref(variable)
        self._creator = None
        self._data = None
        self._generation = 0
        self.name = name

        vdata = variable.data
        self._set_data_type(vdata)

    @property
    def creator(self) -> Optional[function.Function]:
        return self._creator

    @creator.setter
    def creator(self, func: function.Function) -> None:
        self._creator = func
        self._generation = func.generation + 1

    @property
    def data(self) -> Optional[npt.NDArray]:
        """variableに対応するdata配列

        dataがない場合には, ``None``が返される
        """
        return self._data

    @data.setter
    def data(self, d: Optional[npt.NDArray]) -> None:
        self._data = d
        if d is not None:
            self._set_data_type(d)

    @property
    def grad(self) -> Optional[Variable]:
        """variableに対応する勾配"""
        var = self.get_variable()
        return None if var is None else var.grad

    @property
    def generation(self) -> int:
        return self._generation

    def get_variable(self) -> redezero.Variable:
        """対応する:class:`Variable`オブジェクトを返却する

        VariableNodeオブジェクトはvariableオブジェクトへの弱参照を保持する.
        弱参照が有効な場合, このプロパティによってvariableオブジェクトを返却できる.
        弱参照が無効な場合には, VariableNodeオブジェクトの情報から新たなvariableオブジェクトを作成してその値を返却する

        Returns
        -------
        ~redezero.Variable
            VariableNodeオブジェクトが参照するvariableオブジェクト
        """
        var = self._variable()
        if var is not None:
            return var

        var = Variable(self._data, name=self.name)
        return var

    def get_variable_or_none(self) -> Optional[redezero.Variable]:
        """:class:`~redezero.Variable` または ``None``を返却する

        VariableNodeオブジェクトはvariableオブジェクトへの弱参照を保持している.
        弱参照が有効な場合, variableオブジェクトを返却する.
        弱参照が無効な場合, ``None``を返却する

        Returns
        -------
        Optional[~redezero.Variable]
            VariableNodeオブジェクトを参照しているvariableオブジェクト
        """
        return self._variable()

    def retain_data(self) -> None:
        """variable nodeがvariableの対応するdata配列への参照を保持する

        variableへの弱参照がなくなっている場合にはエラーを発生させる
        """
        variable = self._variable()
        if variable is not None:
            self.data = variable.data
        else:
            raise RuntimeError('cannot retain variable data: the variable has already been released.')

    def _set_grad_if_available(self, g: Optional[Variable]) -> None:
        var = self._variable()
        if var is not None:
            var._grad = g

    def _set_data_type(self, d: Optional[npt.NDArray]) -> None:
        if d is not None:
            self.dtype = d.dtype
            self.ndim = d.ndim
            self.size = d.size
            self.shape = d.shape


class Variable:
    """計算をたどるための構造を持つ配列

    全てのvariableは:class:`numpy.ndarray`のdata配列を持つ.

    Attributes
    ----------
    data : Optional[numpy.ndarray]
        データの初期値の配列
    name : Optional[str]
        変数名
    grad : Optional[redezero.Variable]
        勾配の初期値の配列
    creator : Optional[redezero.Variable]
        redezero.Variableインスタンスの生成元redezero.Function
        (逆伝播時に出力から入力をたどるために必要)
    generation : int
        順伝播の世代
        (逆伝播時に順番に伝播するために必要)

    Notes
    -----
    Variableでは次の演算が定義されている.

    Addition :
        ``a + b`` (:meth:`__add__`, :meth:`__radd__`)
    Subtraction :
        ``a - b`` (:meth:`__sub__`, :meth:`__rsub__`)
    Multiplication :
        ``a * b``  (:meth:`__mul__`, :meth:`__rmul__`)
    Division :
        ``a / b`` (:meth:`__div__`, :meth:`__rdiv__`,
                   :meth:`__truediv__`, :meth:`__rtruediv__`)
    Exponentiation :
        ``a ** b`` (:meth:`__pow__`, :meth:`__rpow__`)
    Negation (Arithparams
        ``- a`` (:meth:`__neg__`)
    """
    __array_priority__ = 200

    # 演算子オーバーロード
    __add__ = basic_math.add
    __radd__ = basic_math.add
    __mul__ = basic_math.mul
    __rmul__ = basic_math.mul
    __neg__ = basic_math.neg
    __sub__ = basic_math.sub
    __rsub__ = basic_math.rsub
    __truediv__ = basic_math.div
    __rtruediv__ = basic_math.rdiv
    __pow__ = basic_math.pow

    _data: Optional[npt.NDArray]
    _node: VariableNode
    _grad: Optional[Variable]
    shape: tuple[int, ...]
    ndim: int
    size: int
    dtype: np.dtype

    def __init__(self, data: Optional[npt.NDArray], name=None) -> None:
        """Variableインスタンスの初期化

        Parameters
        ----------
        data : Optional[npt.NDArray]
            :class:`Variable`のdocstringを参照
        name : Optional[str]
            :class:`Variable`のdocstringを参照

        Raises
        ------
        TypeError
            dataの型が:class:`numpy.ndarray`以外は受け付けない
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported.')

        self._data = data
        self._grad = None
        self._set_data_type(data)
        self._node = VariableNode(self, name)

    @property
    def data(self) -> Optional[npt.NDArray]:
        return self._data

    @data.setter
    def data(self, data: Optional[npt.NDArray]) -> None:
        self._data = data

    @property
    def grad(self) -> Optional[Variable]:
        return self._grad

    @grad.setter
    def grad(self, g: Optional[Variable]) -> None:
        self._grad = g

    @property
    def name(self) -> Optional[str]:
        return self._node.name

    @name.setter
    def name(self, n: Optional[str]) -> None:
        self._node.name = n

    @property
    def creator(self) -> Optional[function.Function]:
        return self._node.creator

    @creator.setter
    def creator(self, func: function.Function) -> None:
        self._node.creator = func

    @property
    def generation(self) -> int:
        return self._node.generation

    @property
    def node(self) -> VariableNode:
        return self._node

    def __len__(self) -> int:
        """配列の最初の次元の要素数を返却

        Returns
        --------------
        int
            対象の配列の最初の次元の要素数

        Examples
        --------------
        >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> len(x)
        2
        """
        if self.data is None:
            raise TypeError("len() of unsized object")
        return len(self.data)

    def __repr__(self):
        """インスタンスの中身のデータを出力

        Returns
        --------------
        str
            Variableインスタンスの中のデータ

        Examples
        --------------
        >>> x = Variable(np.array([1, 2, 3]))
        >>> print(x)
        variable([1 2 3])
        >>> x = Variable(None)
        >>> print(x)
        variable(None)
        >>> x = Variable([[1, 2, 3], [4, 5, 6]])
        >>> print(x)
        variable([[1 2 3]
                  [4 5 6]])
        """
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    def _set_data_type(self, d: Optional[npt.NDArray]) -> None:
        if d is not None:
            self.dtype = d.dtype
            self.ndim = d.ndim
            self.size = d.size
            self.shape = d.shape

    def cleargrad(self) -> None:
        """微分値のリセット
        """
        self._grad = None

    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
        """variableインスタンスに対して誤差逆伝播を実施

        参考: Chainer variable.py
        https://github.com/chainer/chainer/blob/v5/chainer/variable.py

        Parameters
        ----------
        retain_grad : bool
            変数が勾配を保持するかのフラグ
        create_graph: bool
            逆伝播で行う計算に対してさらに逆伝播を行うかのフラグ
        """
        with configuration.using_config('enable_backprop', create_graph):
            self._backward_main(retain_grad)

    def _backward_main(self, retain_grad: bool) -> None:
        if self.creator is None:
            return

        if self.grad is None:
            # y.gradの微分値を設定
            self.grad = Variable(np.ones_like(self.data))

        funcs: list = []
        # funcsリストに同じ関数を重複して追加することを防ぐために使用
        # (関数のbackwardメソッドが誤って複数回呼ばれることを防ぐ)
        seen_set: set[function.Function] = set()
        # VariableNodeに対応する勾配を保持
        grads = _backprop_utils.GradTable(load_if_new=True)
        grads[self._node] = self._grad

        def add_func(f: function.Function) -> None:
            """逆伝播対象の関数追加

            逆伝播対象の関数を追加するたびに世代の昇順ソート実施

            Parameters
            ----------
            f : ~redezero.Function]
                逆伝播対象のFunction
            """
            if f not in seen_set:
                # heapqは最小値を取り出す仕組みのため, generationを負数に変換
                # Chainer _backprop.pyの下記処理を参考
                # https://github.com/chainer/chainer/blob/536cda7c9a146b9198f83837ba439a5afbdc074d/chainer/_backprop.py#L161
                heapq.heappush(funcs, (-f.generation, len(seen_set), f))
                seen_set.add(f)
        add_func(self.creator)
        leaf_nodes: set[VariableNode] = set()

        # 逆伝播のメイン処理
        # 出力から順にたどる関数がなくなるまで計算
        while funcs:
            f: function.Function = heapq.heappop(funcs)[2]
            inputs = f.inputs
            target_input_indexes = tuple([i for i in range(len(inputs))])

            # y_nodeはweakref
            outputs = [y_node() for y_node in f.outputs]
            # 順伝播の出力変数が複数ある場合, 逆伝播を呼び出していない出力変数の勾配は`None`が返却される
            out_grad = tuple([grads.pop(y_node) for y_node in outputs])
            if not target_input_indexes:
                continue

            # 現在の入力勾配を収集する
            target_inputs = tuple([inputs[i] for i in target_input_indexes])
            in_grad: dict[redezero.VariableNode, list[redezero.Variable]] = {}
            for x in target_inputs:
                if x not in in_grad:
                    in_grad[x] = grads.get_as_list(x)
                    # 勾配のリセット（不要なメモリ確保を防ぐため）
                    x._set_grad_if_available(None)

            _backprop_utils.backprop_step(f, target_input_indexes, out_grad, in_grad)

            # 途中変数の勾配が不要な場合にはリセット（不要なメモリ確保を防ぐため）
            for y, gy in zip(outputs, out_grad):
                if y is not None and y is not self.node:
                    y._set_grad_if_available(gy if retain_grad else None)
            del gy, out_grad

            # 変数の親ノードの設定
            for x, gx in in_grad.items():
                if not gx:
                    continue
                if x.creator is None:
                    # 生成元関数がない場合, 自身を末端ノードとして追加
                    leaf_nodes.add(x)
                else:
                    add_func(x.creator)
            del gx, in_grad

        # 末端ノードの勾配更新
        for x in leaf_nodes:
            x_var = x.get_variable_or_none()
            x_grad = grads.pop(x)
            if x_var is not None:
                x_var._grad = x_grad
        grads.assert_no_grad()

    def reshape(self, *shape: int) -> Variable:
        """配列の形状を変更する

        Parameters
        ----------
        *shape : int
            変換したい配列の形状

        Returns
        ----------
        ~redezero.Variable
            形状を変更したVariableインスタンス

        Examples
        -----------
        >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> y = x.reshape((6,))
        >>> y
        variable([1 2 3 4 5 6])
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return redezero.functions.reshape(self, shape)

    def transpose(self, *axes: int) -> Variable:
        """行列の転置を行う

        Parameters
        ----------
        *axes : int
            転置を行う行列の軸

        Returns
        ----------
        ~redezero.Variable
            転置後のVariableインスタンス

        Examples
        -----------
        >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> y = x.transpose()
        >>> y
        variable([[1 2]
                  [3 4]
                  [5 6]])
        """
        _axes: Optional[Sequence[int]] = axes
        if len(axes) == 0:
            _axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                _axes = axes[0]

        return redezero.functions.transpose(self, _axes)

    @property
    def T(self) -> redezero.Variable:
        """行列の転置
        """
        return redezero.functions.transpose(self)

    def retain_data(self) -> None:
        """対応するVariableNodeがdata配列を参照できるようにする"""
        self._node.data = self._data


def as_variable(obj) -> Variable:
    """variableを:class:`redezero.Variable`に変換

    Parameters
    ----------
    obj : object
        変換対象のオブジェクト

    Returns
    -------
    redezero.Variable
        入力値をVariableに変換した値
    """
    if isinstance(obj, Variable):
        return obj

    return Variable(obj)
