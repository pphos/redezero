import pytest
import numpy as np

from redezero import function
from redezero import Variable


class TestVariableNode:
    def test_正常系_インスタンス初期化(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]), name='x')

        assert x.node.creator is None
        assert x.node.data is None
        assert x.generation == 0
        assert x.name == 'x'
        assert x.node.dtype == x.dtype
        assert x.node.shape == x.shape

        # Variableオブジェクトの取得
        var = x.node.get_variable()
        assert var == x

    def test_正常系_ノードの生成元関数の設定(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        f = function.Function()
        f.generation = 0

        x.node.creator = f
        assert x.node.generation == 1

    def test_正常系_variableのdata配列への参照の保持(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))

        x.node.retain_data()
        assert np.array_equal(x.node.data, x.data)

    def test_異常系_variableへの弱参照が無効(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        x_node = x.node
        # VariableNodeからVariableへの参照を切るためにxを削除
        del x

        with pytest.raises(RuntimeError) as e:
            x_node.retain_data()
        assert str(e.value) == 'cannot retain variable data: the variable has already been released.'


class TestVariable:
    def params_for_property():
        """test_正常系_プロパティ動作のパラメータ
        """
        return {
            'dataが0次元': (np.array(1)),
            'dataが1次元': (np.array([1, 2])),
            'dataが2次元': (np.array([[1, 2, 3], [4, 5, 6]]))
        }

    @pytest.mark.parametrize(
        'x',
        params_for_property().values(),
        ids=params_for_property().keys()
    )
    def test_正常系_プロパティ動作(self, x):
        y = Variable(x)
        # プロパティのテスト
        assert np.array_equal(y.data, x)
        assert y.grad is None
        assert y.generation == 0
        assert y.shape == x.shape
        assert y.ndim == x.ndim
        assert y.size == x.size
        assert y.dtype == x.dtype

    def params_for_data_is_not_ndarray():
        """test_異常系_dataがndarrayでないのパラメータ
        """
        return {
            'dataがint型': (1),
            'dataがfloat型': (1.0)
        }

    @pytest.mark.parametrize(
        'x',
        params_for_data_is_not_ndarray().values(),
        ids=params_for_data_is_not_ndarray().keys()
    )
    def test_異常系_dataがndarrayでない(self, x):
        with pytest.raises(TypeError) as e:
            Variable(x)

        assert str(e.value) == f'{type(x)} is not supported.'

    def test_正常系_len動作(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        assert len(x) == 2

    def params_for_print():
        """正常系_print動作のパラメータ
        """
        return {
            '引数がNone': (
                None,
                'variable(None)\n'
            ),
            '引数が1次元': (
                np.array([1, 2, 3]),
                'variable([1 2 3])\n'
            ),
            '引数が2次元': (
                np.array([[1, 2, 3], [4, 5, 6]]),
                f"variable([[1 2 3]\n {' ' * 9}[4 5 6]])\n"
            )
        }

    @pytest.mark.parametrize(
        'x, msg',
        params_for_print().values(),
        ids=params_for_print().keys()
    )
    def test_正常系_print動作(self, capfd, x, msg):
        y = Variable(x)
        print(y)
        out, _ = capfd.readouterr()
        assert out == msg

    def params_for_backward():
        """正常系_逆伝播のパラメータ
        """
        return {
            '中間変数が勾配を保持しない': (False,),
            '中間変数が勾配を保持する': (True,)
        }

    @pytest.mark.parametrize(
        'retain_grad',
        params_for_backward().values(),
        ids=params_for_backward().keys()
    )
    def test_正常系_逆伝播_一つの関数が複数同じ変数を参照する(self, retain_grad):
        # ゼロから作るDeep Learning 3 P107参照
        x = Variable(np.array(2.0))
        expect = np.array(64.0)

        # 演算
        a = x ** 2
        b = a ** 2
        c = a ** 2
        y = b + c

        # 逆伝播の実施
        y.backward(retain_grad=retain_grad)
        assert x.grad.data == expect

        # 中間変数の勾配確認
        middles = [a, b, c]
        for middle in middles:
            if retain_grad:
                assert middle.grad is not None
            else:
                assert middle.grad is None

    def test_正常系_逆伝播_高階微分(self):
        x = Variable(np.array(3.0))
        y = x ** 3
        expects = {
            'gx': np.array(27.0),
            'gx2': np.array(18.0)
        }

        # 逆伝播の実施
        y.backward(create_graph=True)
        gx = x.grad
        assert gx.data == expects['gx']

        # 2階微分
        x.cleargrad()
        gx.backward()
        gx2 = x.grad
        assert gx2.data == expects['gx2']

    def test_正常系_勾配のリセット(self):
        x = Variable(np.array(3.0))
        expect = np.array(3.0)

        # 1回目の計算
        y1 = 2 * x
        y1.backward()

        # 勾配のリセット
        x.cleargrad()
        y2 = 3 * x
        y2.backward()
        assert x.grad.data == expect

    def params_for_reshape():
        """正常系_reshape動作のパラメータ
        """
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        expect = (3, 2)

        return {
            '引数がタプル': (x, (3, 2), expect),
            '引数がリスト': (x, [3, 2], expect),
        }

    @pytest.mark.parametrize(
        'x, shape, expect',
        params_for_reshape().values(),
        ids=params_for_reshape().keys()
    )
    def test_正常系_reshape動作(self, x, shape, expect):
        # 順伝播の確認
        y = x.reshape(shape)
        assert y.shape == expect

        # 逆伝播の確認
        y.backward()
        assert x.grad.shape == x.shape

    def params_for_transpose():
        """正常系_transpose動作のパラメータ
        """
        x = Variable(np.random.randn(1, 2, 3, 4))

        return {
            'axesがNone': (x, None, (4, 3, 2, 1)),
            'axesがタプル': (x, (1, 0, 3, 2), (2, 1, 4, 3)),
            'axesがリスト': (x, [1, 0, 3, 2], (2, 1, 4, 3))
        }

    @pytest.mark.parametrize(
        'x, axes, expect',
        params_for_transpose().values(),
        ids=params_for_transpose().keys()
    )
    def test_正常系_transpose動作(self, x, axes, expect):
        # 順伝播の確認
        y = x.transpose(axes)
        assert y.shape == expect

        # 逆伝播の確認
        y.backward()
        assert x.grad.shape == x.shape

    def test_正常系_transpose動作_引数が可変長(self):
        x = Variable(np.random.randn(1, 2, 3, 4))

        # 順伝播の確認
        y = x.transpose(1, 0, 3, 2)
        assert y.shape == (2, 1, 4, 3)

        # 逆伝播の確認
        y.backward()
        assert x.grad.shape == x.shape
