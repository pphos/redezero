from __future__ import annotations
from typing import cast
from weakref import ref, ReferenceType
import numpy as np

from redezero import types
from redezero import configuration
from redezero import variable


class Function:
    """微分可能関数のインターフェース

    `Function`クラスのサブクラスは`forward`, `backward`メソッドを実装することで順伝播の計算と逆伝播の自動導出ができます

    Attributes
    ----------
    inputs : Variable | list[Variable]
        関数の入力変数
    outputs : Variable | list[Variable]
        関数の出力変数
    generation : int
        順伝播時の世代数
    """
    inputs: tuple[variable.Variable, ...]
    outputs: list[ReferenceType[variable.Variable]]

    def apply(self, inputs: tuple[types.OperandValue, ...]) -> list[variable.Variable]:
        """出力変数の計算を行い計算グラフへ追加

        基本的な振る舞いは:class:`FunctionNode`のドキュメントに記載されています

        Parameters
        ----------
        inputs : tuple[~redezero.Variable | np.ndarray, ...]
            関数の入力変数

        Returns
        -------
        tuple[~redezero.Variable] | ~redezero.Variable
            関数の出力変数
        """
        variable_inputs = tuple([variable.as_variable(x) for x in inputs])

        xs = [x.data for x in variable_inputs]
        ys = self.forward(*xs)
        tupled_ys = ys if isinstance(ys, tuple) else (ys,)
        outputs = [variable.Variable(y) for y in tupled_ys]

        if configuration.Config.enable_backprop:
            self.generation: int = max([x.generation for x in variable_inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = variable_inputs
            # FunctionとVariable間の循環参照を解消するために弱参照モジュール(weakref)を使用
            # (弱参照は、参照カウントを増やさずに別オブジェクトを参照する機能)
            self.outputs = [ref(output) for output in outputs]

        return outputs

    def forward(self, xs) -> tuple[np.ndarray, ...]:
        """入力配列に対する順伝播の実施

        Parameters
        ----------
        xs : tuple[numpy.ndarray]
            順伝播を適用する入力配列

        Returns
        -------
        tuple[numpy.ndarray, ...]
            順伝播適用後の出力値のタプル
        """
        raise NotImplementedError()

    def backward(self, gys) -> tuple[variable.Variable, ...]:
        """勾配配列に対する逆伝播の実施

        Parameters
        ----------
        gys : tuple[Variable]
            逆伝播を適用する勾配配列

        Returns
        -------
        tuple[Variable]
            逆伝播適用後の出力勾配タプル
        """
        raise NotImplementedError()