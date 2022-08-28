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

    参照: Chainer function_node.py
    https://github.com/chainer/chainer/blob/238b1c0978f506a8c35b685ebd927b9885b648b3/chainer/function_node.py

    `Function`クラスのサブクラスは`forward`, `backward`メソッドを実装することで順伝播の計算と逆伝播の自動導出が

    入出力variableと計算グラフ内の対応するvariable nodeは区別される.
    Functionはoutputsとして:class:`Variable`オブジェクトを返す:class:`Variable`の関数として機能するが,
    これらのオブジェクトは直接計算グラフに追加されない.

    Attributes
    ----------
    inputs : tuple[~redezero.VariableNode, ...]
        関数の入力変数
    outputs : tuple[ReferenceType[~redezero.VariableNode], ...]
        関数の出力変数
    generation : int
        順伝播時の世代数
    """
    _input_indexes_to_retain: Optional[Iterable[int]]
    _output_indexes_to_retain: Optional[Iterable[int]]
    inputs: tuple[redezero.VariableNode, ...]
    outputs: tuple[ReferenceType[redezero.VariableNode], ...]
    generation: int

    def apply(self, inputs: tuple[types.OperandValue, ...]) -> tuple[redezero.Variable, ...]:
        """出力変数の計算を行い計算グラフへ追加

        基本的な振る舞いは:class:`Function`のドキュメントに記載されています

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

        # 順伝播を行い, Variableオブジェクトに変換
        self._input_indexes_to_retain = None
        self._output_indexes_to_retain = None
        ys = self.forward(xs)
        outputs = tuple([variable.Variable(y) for y in ys])

        if configuration.Config.enable_backprop:
            self.generation: int = max([x.generation for x in variable_inputs])
            for output in outputs:
                output.creator = self
            self.inputs = tuple([x.node for x in variable_inputs])
            # FunctionとVariable間の循環参照を解消するために弱参照モジュール(weakref)を使用
            # (弱参照は、参照カウントを増やさずに別オブジェクトを参照する機能)
            self.outputs = tuple([ref(output.node) for output in outputs])

            if self._input_indexes_to_retain is not None:
                for index in self._input_indexes_to_retain:
                    variable_inputs[index].retain_data()

            if self._output_indexes_to_retain is not None:
                retained_data = []
                for index in self._output_indexes_to_retain:
                    outputs[index].retain_data()
                    retained_data.append(outputs[index])

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

    def backward(self, target_input_indexes: tuple[int, ...],
                 gys: tuple[redezero.Variable, ...]) -> tuple[redezero.Variable, ...]:
        """勾配配列に対する逆伝播の実施

        Parameters
        ----------
        target_input_indexes : tuple[int, ...]
            逆伝播で勾配が必要な入力変数のインデックス
        gys : tuple[~redezero.Variable, ...]
            逆伝播を適用する勾配配列

        Returns
        -------
        tuple[~redezero.Variable, ...]
            逆伝播適用後の出力勾配タプル
        """
        raise NotImplementedError()

    def backward_accumulate(self, target_input_indexes: tuple[int, ...],
                            grad_outputs: tuple[redezero.Variable, ...],
                            grad_inputs: tuple[redezero.Variable, ...]) -> tuple[redezero.Variable, ...]:
        """入力変数に関する勾配を計算し, 勾配を蓄積する

        参照: Chainer function_node.py
        https://github.com/chainer/chainer/blob/238b1c0978f506a8c35b685ebd927b9885b648b3/chainer/function_node.py

        同じ変数に複数の関数が適用される場合の逆伝播の計算と勾配の蓄積を合わせた処理を行う

        Parameters
        ----------
        target_input_indexes : tuple[int, ...]
            勾配計算が必要な入力変数
        grad_outputs : tuple[~redezero.Variable, ...]
            出力変数に関する勾配
        grad_inputs : tuple[~redezero.Variable, ...]
            ``target_input_indexes``で指定した入力変数に関する勾配

        Returns
        -------
        tuple[~redezero.Variable]
            入力変数に関する勾配
            戻り値のタプルは``target_input_indexes``と同じ形状である

        Notes
        -----
        同じ変数が関数の入力変数として複数渡された場合, ``grad_inputs``の最初の位置にのみ,
        入力変数に対応する勾配を含めることができ, そのほかの値は``None``とする.
        """
        gxs = self.backward(target_input_indexes, grad_outputs)

        len_gxs = len(gxs)
        if len_gxs == len(self.inputs):
            gxs = tuple([gxs[i] for i in target_input_indexes])
        elif len_gxs != len(target_input_indexes):
            raise ValueError('number of returned values is incorrect.')

        ret = []
        for gx, g_input in zip(gxs, grad_inputs):
            if g_input is None:
                ret.append(gx)
            elif gx is None:
                ret.append(g_input)
            else:
                ret.append(gx + g_input)

        return tuple(ret)

    def retain_inputs(self, indexes: Iterable[int]) -> None:
        """variable nodeが入力data配列を保持できるようにする

        :meth:`forward`からこの関数を呼び出すことで, 逆伝播に必要な入力配列を指定できる.
        このメソッドが呼び出されない場合, メソッドはすべての入力配列を保持する.
        全ての入力配列を解放したい場合, `()`を指定してこのメソッドを呼び出す.

        Notes
        -----
        :meth:`forward()`の外部から呼び出してはいけない

        Parameters
        ----------
        indexes : Iterable[int]
            逆伝播を必要としないvariableのインデックスのイテレータ
        """
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes: Iterable[int]) -> None:
        """variable nodeが出力data配列を保持できるようにする

        :meth:`forward`からこの関数を呼び出すことで, 逆伝播に必要な入力配列を指定できる.
        このメソッドが呼び出されない場合, 出力variableのdata配列は保持されない.
        出力は:attr:`output_data`に保持される.

        Notes
        -----
        :meth:`backward()`の外部から呼び出してはいけない

        Parameters
        ----------
        indexes : Iterable[int]
            逆伝播を必要としないvariableのインデックスのイテレータ
        """
        self._output_indexes_to_retain = indexes

    def get_retained_inputs(self) -> tuple[redezero.Variable, ...]:
        """入力variableを返却する

        :meth:`forward`で保持された入力variableを抽出するために利用される

        Returns
        ----------
        tuple[~redezero.Variable, ...]
            入力variableのタプル
        """
        inputs = self.inputs
        return tuple([inputs[index].get_variable()
                      for index in self._input_indexes_to_retain])

    def get_retained_outputs(self) -> tuple[redezero.Variable, ...]:
        """出力variableを返却する

        :meth:`forward`で保持された出力variableを抽出するために利用される

        Returns
        -------
        tuple[~redezero.Variable, ...]
            出力variableのタプル
        """
        outputs = []
        for index in self._output_indexes_to_retain:
            if (output := self.outputs[index]()) is not None:
                outputs.append(output.get_variable())
            else:
                raise RuntimeError('cannnot retain variable data: the variable has already been released.')

        return tuple(outputs)
