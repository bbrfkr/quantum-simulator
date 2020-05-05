"""
baseパッケージ内で利用する定数群
"""

import numpy as np
from typing import List

# 計算時の近似桁数
RELATIVE_TOLERANCE = 1.0e-5
ABSOLUTE_TOLERANCE = 1.0e-8
AROUNDED_DECIMALS = 5


def isclose(a: np.array, b: np.array) -> np.array:
    """np.iscloseの本プロジェクト用のラッパー"""
    return np.isclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def allclose(a: np.array, b: np.array) -> bool:
    """np.allcloseの本プロジェクト用のラッパー"""
    return np.allclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def around(a: np.array) -> np.array:
    """np.aroundの本プロジェクト用のラッパー"""
    return np.around(a, AROUNDED_DECIMALS)


def is_pow2(a: int) -> bool:
    """整数が正の2の累乗であるか判定する"""
    # 2より小さい値の場合は偽
    if a < 2:
        return False

    # ビットの論理積で計算する
    return not a & (a - 1)


def is_probabilities(target_list: List[float]):
    """与えられたfloatリストが確率分布であるか判定する"""

    total = 0
    for element in target_list:
        if element < 0:
            return False
        total += element
        if round(total, AROUNDED_DECIMALS) > 1.0:
            return False

    if round(total, AROUNDED_DECIMALS) < 1.0:
        return False

    return True
