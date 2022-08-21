from __future__ import annotations
from pathlib import Path
from typing import Optional
import subprocess

import redezero
from redezero import function


def _dot_var(v: redezero.Variable, verbose: bool = False) -> str:
    """VariableをDOT言語の文字列へ変換する

    Parameters
    ----------
    v : Variable
        DeZeroの変数
    verbose : bool, optional
        ndarrayインスタンスの「形状」と「型」も合わせて出力するフラグ

    Returns
    -------
    str
        Variableインスタンスの情報をDOT言語に変換した文字列
    """
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += f'{str(v.shape)} {str(v.dtype)}'

    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'


def _dot_func(f: function.Function) -> str:
    """FunctionをDOT言語の文字列へ変換する

    Parameters
    ----------
    f : Function
        DeZeroの関数

    Returns
    -------
    str
        Functionインスタンスの情報をDOT言語に変換した文字列
    """
    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # yはweakref

    return txt


def get_dot_graph(output: redezero.Variable, verbose: bool = True) -> str:
    """計算グラフからDOT言語へ変換する

    Parameters
    ----------
    output : Variable
        計算グラフの最終出力変数
    verbose : bool, optional
        ndarrayインスタンスの「形状」と「型」も合わせて出力するフラグ

    Returns
    -------
    str
        計算グラフの情報をDOT言語に変換した文字列
    """
    txt = ''
    funcs: list[function.Function] = []
    seen_set = set()

    def add_func(f: Optional[function.Function]) -> None:
        """関数の追加
        """
        if (f not in seen_set) and (f is not None):
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output: redezero.Variable, verbose: bool = True, to_file: str = 'graph.png') -> None:
    """計算グラフをGraphvizで画像に変換する

    Parameters
    ----------
    output : Variable
        計算グラフの最終出力変数
    verbose : bool, optional
        ndarrayインスタンスの「形状」と「型」も合わせて出力するフラグ
    to_file : str, optional
        保存する画像のファイル名
    """
    # 計算グラフをDOT言語に変換
    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    tmp_dir = Path('.dezero')
    if not tmp_dir.exists():
        tmp_dir.mkdir()
    graph_path = Path(tmp_dir).joinpath('tmp_graph.dot')
    Path(graph_path).write_text(dot_graph)

    # osのdotコマンド呼び出し
    extension = Path(to_file).suffix.lstrip('.')
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)
