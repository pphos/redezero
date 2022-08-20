import numpy as np

from redezero import Variable
from redezero import functions as F


def test_正常系_tanh関数動作():
    x = Variable(np.array(0.5))
    expects = {
        'x_grad': np.array(0.78644773296),
        'y_data': np.array(0.46211715726)
    }

    # 順伝播の確認
    y = F.tanh(x)
    assert np.isclose(y.data, expects['y_data'])

    # 逆伝播の確認
    y.backward(create_graph=True)
    assert np.isclose(x.grad.data, expects['x_grad'], atol=1e-10)
