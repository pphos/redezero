import pytest
import numpy as np

from redezero import Variable


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
