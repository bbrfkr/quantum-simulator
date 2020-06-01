import importlib
import os

env_use_cupy = os.environ.get("USE_CUPY")
use_cupy = False if env_use_cupy != "True" else True


def is_cupy():
    """
    cupyが有効か否かを返す関数

    Returns:
        bool: cupyが有効か
    """
    return use_cupy


def xp_factory():
    """
    numpyとcupyを動的インポートする関数

    Returns:
        object: 動的インポートしたモジュールオブジェクト
    """
    if is_cupy():
        return importlib.import_module("cupy")
    else:
        return importlib.import_module("numpy")
