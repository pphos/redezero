import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系_reshape動作():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    expect_y = (6,)

    # 順伝播の確認
    y = F.reshape(x, (6,))
    assert y.shape == expect_y

    # 逆伝播の確認
    y.backward()
    assert x.grad.shape == x.shape
