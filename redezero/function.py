from __future__ import annotations
from typing import Iterable, Optional
from weakref import ref, ReferenceType
import contextlib
import numpy.typing as npt

from redezero import types
from redezero import configuration
from redezero import variable
import redezero


def no_grad() -> contextlib._GeneratorContextManager[None]:
    """逆伝播の無効化

    using_config('enable_backprop', False)のショートハンド
    """
    return configuration.using_config('enable_backprop', False)


class Function:
    """微分可能関数のインターフェース

    `Function`クラスのサブクラスは`forward`, `backward`メソッドを実装することで順伝播の計算と逆伝播の自動導出ができる

    Attributes
    ----------
    inputs : Variable | list[Variable]
        関数の入力変数
    outputs : Variable | list[Variable]
        関数の出力変数
    generation : int
        順伝播時の世代数
    """
    inputs: tuple[redezero.Variable, ...]
    outputs: tuple[ReferenceType[redezero.Variable], ...]
    generation: int

    def apply(self, inputs: tuple[types.OperandValue, ...]) -> tuple[redezero.Variable, ...]:
        """出力変数の計算を行い計算グラフへ追加

        基本的な振る舞いは:class:`FunctionNode`のドキュメントに記載されています

        Parameters
        ----------
        inputs : tuple[~redezero.Variable | numpy.ndarray, ...]
            関数の入力変数

        Returns
        -------
        tuple[~redezero.Variable, ...]
            関数の出力変数
        """
        variable_inputs = tuple([variable.as_variable(x) for x in inputs])
        xs = tuple([x.data for x in variable_inputs])
        ys = self.forward(xs)
        outputs = tuple([variable.Variable(y) for y in ys])

        if configuration.Config.enable_backprop:
            self.generation: int = max([x.generation for x in variable_inputs])
            for output in outputs:
                output.creator = self
            self.inputs = variable_inputs
            # FunctionとVariable間の循環参照を解消するために弱参照モジュール(weakref)を使用
            # (弱参照は、参照カウントを増やさずに別オブジェクトを参照する機能)
            self.outputs = tuple([ref(output) for output in outputs])

        return outputs

    def forward(self, xs: tuple[npt.NDArray, ...]) -> tuple[npt.NDArray, ...]:
        """入力配列に対する順伝播の実施

        Parameters
        ----------
        xs : tuple[numpy.ndarray, ...]
            順伝播を適用する入力配列

        Returns
        -------
        tuple[numpy.ndarray, ...]
            順伝播適用後の出力値のタプル
        """
        raise NotImplementedError()

    def backward(self, xs: tuple[redezero.Variable, ...],
                 gys: tuple[redezero.Variable, ...]) -> tuple[redezero.Variable, ...]:
        """勾配配列に対する逆伝播の実施

        Parameters
        ----------
        xs : tuple[~redezero.Variable, ...]
            逆伝播を適用する入力配列
        gys : tuple[~redezero.Variable, ...]
            逆伝播を適用する勾配配列

        Returns
        -------
        tuple[~redezero.Variable, ...]
            逆伝播適用後の出力勾配タプル
        """
        raise NotImplementedError()
