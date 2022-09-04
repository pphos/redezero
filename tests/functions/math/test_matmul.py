import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系_matmul動作():
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))

    y = F.matmul(x, W)
    assert y.shape == (2, 4)

    y.backward()

    assert x.grad.shape == (2, 3)
    assert W.grad.shape == (3, 4)
