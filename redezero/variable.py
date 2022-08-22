from __future__ import annotations
from typing import Optional, Sequence
import heapq
import numpy as np
import numpy.typing as npt

from redezero import configuration
from redezero import function
import redezero
from redezero.functions.math import basic_math


class Variable:
    """計算をたどるための構造を持つ配列

    全てのvariableは:class:`numpy.ndarray`のdata配列を持つ.

    Attributes
    ----------
    data : numpy.ndarray
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

    data: npt.NDArray
    name: Optional[str]
    grad: Optional[Variable]
    creator: Optional[function.Function]
    generation: int

    def __init__(self, data: npt.NDArray, name=None) -> None:
        """Variableインスタンスの初期化

        Parameters
        ----------
        data : npt.NDArray
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

        self.data: npt.NDArray = data
        self.name: Optional[str] = name
        self.grad: Optional[Variable] = None
        self.creator: Optional[function.Function] = None
        self.generation: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        """配列の形状

        Returns
        -------
        tuple[int]
            配列の次元数のタプル

        Examples
        --------
        >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> x.shape
        (2, 3)
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """配列の次元数

        Returns
        -------
        int
            配列の次元数

        Example
        -------
        >>> x = Variable(np.array([1, 2, 3]))
        >>> x.ndim
        1
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        """配列の要素数

        Returns
        -------
        int
            配列の要素数

        Examples
        --------
        >>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        >>> x.size
        6
        """
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """データ型の表示

        Returns
        -------
        np.dtype
            配列の要素のデータ型

        Examples
        --------
        >>> x = Variable(np.array([1, 2, 3]))
        >>> x.dtype
        dtype('int64')
        """
        return self.data.dtype

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

    def set_creator(self, func: function.Function) -> None:
        """生成元Functionのつながりを保持

        Parameters
        --------------
        func : redezero.Function
            Variableインスタンスの生成元Function
        """
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self) -> None:
        """微分値のリセット
        """
        self.grad = None

    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
        """variableインスタンスに対して誤差逆伝播を実施

        Parameters
        --------------
        retain_grad : bool
            変数が勾配を保持するかのフラグ
        create_graph: bool
            逆伝播で行う計算に対してさらに逆伝播を行うかのフラグ
        """
        if self.grad is None:
            # y.gradの微分値を設定
            self.grad = Variable(np.ones_like(self.data))

        funcs: list = []
        # funcsリストに同じ関数を重複して追加することを防ぐために使用
        # (関数のbackwardメソッドが誤って複数回呼ばれることを防ぐ)
        seen_set: set[function.Function] = set()

        def add_func(f: Optional[function.Function]) -> None:
            """逆伝播対象の関数追加

            逆伝播対象の関数を追加するたびに世代の昇順ソート実施

            Parameters
            ----------
            f : Optinal[Function]
                逆伝播対象のFunction
            """
            if (f not in seen_set) and (f is not None):
                # heapqは最小値を取り出す仕組みのため, generationを負数に変換
                # Chainer _backprop.pyの下記処理を参考
                # https://github.com/chainer/chainer/blob/536cda7c9a146b9198f83837ba439a5afbdc074d/chainer/_backprop.py#L161
                heapq.heappush(funcs, (-f.generation, len(seen_set), f))
                seen_set.add(f)
        add_func(self.creator)

        # 逆伝播のメイン処理
        # 出力から順にたどる関数がなくなるまで計算
        while funcs:
            f: function.Function = heapq.heappop(funcs)[2]
            gys: list[Variable] = []
            for output_ref in f.outputs:
                if ((output := output_ref()) is not None) and (output.grad is not None):
                    gys.append(output.grad)

            with configuration.using_config('enable_backprop', create_graph):
                gxs = f.backward(f.inputs, tuple(gys))
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # x.grad += gx はコピーを行わずメモリの値を直接上書きするため使用しない
                        # (入力と出力が同じgradが参照されることになってしまう)
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                # 途中の変数の微分値をすべてリセット (不要なメモリ確保を防ぐため)
                for y_ref in f.outputs:
                    if (y := y_ref()) is not None:
                        y.grad = None

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
