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

    def test_正常系len動作(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        assert len(x) == 2
