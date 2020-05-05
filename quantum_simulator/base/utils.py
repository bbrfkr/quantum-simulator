"""
baseパッケージ内で利用する定数群
"""

import numpy as np

# 計算時の近似桁数
RELATIVE_TOLERANCE = 1.0e-5
ABSOLUTE_TOLERANCE = 1.0e-8


def isclose(a: np.array, b: np.array) -> np.array:
    """np.iscloseの本プロジェクト用のラッパー"""
    return np.isclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def allclose(a: np.array, b: np.array) -> bool:
    """np.allcloseの本プロジェクト用のラッパー"""
    return np.allclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)
