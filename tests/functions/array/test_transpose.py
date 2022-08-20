import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系_transpose動作():
    x = Variable(np.random.randn(1, 2, 3, 4))
    expect = (4, 3, 2, 1)

    # 順伝播の確認
    y = F.transpose(x)
    assert y.shape == expect

    # 逆伝播の確認
    y.backward()
    assert x.grad.shape == x.shape
