import contextlib
from typing import Iterator


class Config:
    """設定クラス

    Attributes
    --------------
    enable_backprop : bool
        逆伝播を有効にするかを決めるフラグ
    """
    enable_backprop: bool = True


@contextlib.contextmanager
def using_config(name: str, value: bool) -> Iterator[None]:
    """利用コンフィグの指定

    Parameters
    ----------
    name : str
        コンフィグの設定パラメータ名
    value : bool
        コンフィグの設定値
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
