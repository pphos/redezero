import pytest
import shutil
import filecmp
import numpy as np
from pathlib import Path

from redezero import utils
from redezero import Variable


def params_for_dot_var_node():
    """正常系_Variableインスタンス情報をDOT言語に変換のパラメータ
    """
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]), name='x')

    return {
        '入力変数の形状も出力する': (
            x.node,
            True,
            f'{id(x.node)} [label="x: (2, 3) int64", color=orange, style=filled]\n'
        ),
        '入力変数の形状を出力しない': (
            x.node,
            False,
            f'{id(x.node)} [label="x", color=orange, style=filled]\n'
        )
    }


@pytest.mark.parametrize(
    'x, verbose, expect_msg',
    params_for_dot_var_node().values(),
    ids=params_for_dot_var_node().keys()
)
def test_正常系_Variableインスタンス情報をDOT言語に変換(x, verbose, expect_msg):
    assert utils._dot_var_node(x, verbose=verbose) == expect_msg


def test_正常系_Functionインスタンス情報をDOT言語に変換():
    # Functionインスタンスのセットアップ
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y = x0 + x1

    # 期待するDOT言語の文字列
    expect = f'{id(y.creator)} [label="Add", color=lightblue, style=filled, shape=box]\n'
    expect += f'{id(x0.node)} -> {id(y.creator)}\n'
    expect += f'{id(x1.node)} -> {id(y.creator)}\n'
    expect += f'{id(y.creator)} -> {id(y.node)}\n'

    assert utils._dot_func(y.creator) == expect


def test_正常系_計算グラフからDOT言語に変換():
    x0 = Variable(np.array(1.0), name='x0')
    x1 = Variable(np.array(1.0), name='x1')
    y = x0 + x1
    y.name = 'y'

    # 期待するDOT言語の文字列
    expect = 'digraph g {\n'
    expect += f'{id(y.node)} [label="y", color=orange, style=filled]\n'
    expect += f'{id(y.creator)} [label="Add", color=lightblue, style=filled, shape=box]\n'
    expect += f'{id(x0.node)} -> {id(y.creator)}\n'
    expect += f'{id(x1.node)} -> {id(y.creator)}\n'
    expect += f'{id(y.creator)} -> {id(y.node)}\n'
    expect += f'{id(x0.node)} [label="x0", color=orange, style=filled]\n'
    expect += f'{id(x1.node)} [label="x1", color=orange, style=filled]\n'
    expect += '}'

    assert utils.get_dot_graph(y, verbose=False) == expect


@pytest.fixture
def dot_graph():
    """plot_dot_graph用のfixture
    """
    # SetUp
    output_dir = Path('.dezero')
    img = {
        'actual': output_dir.joinpath('goldstein.png'),
        'expect': Path('tests/resources/image/goldstein.png')
    }
    yield img

    # 生成した計算グラフの削除
    if img['actual'].exists():
        shutil.rmtree(output_dir)


def test_正常系_計算グラフをGraphvizで画像に変換(dot_graph):
    def goldstein(x, y) -> Variable:
        """サンプル用のGolestein-Price関数
        """
        z = (1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * \
            (30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2))
        return z

    # 計算グラフの作成
    x = Variable(np.array(1.0), name='x')
    y = Variable(np.array(1.0), name='y')
    z = goldstein(x, y)
    z.backward()
    z.name = 'z'

    # 計算グラフから画像の生成
    utils.plot_dot_graph(z, verbose=False, to_file=dot_graph['actual'])

    # 出力画像の比較
    assert filecmp.cmp(dot_graph['actual'], dot_graph['expect'])
