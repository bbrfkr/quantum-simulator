"""
baseパッケージ内で利用するユーティリティメソッド群
"""

from typing import List

import numpy as np

# 計算時の近似桁数
RELATIVE_TOLERANCE = 1.0e-5
ABSOLUTE_TOLERANCE = 1.0e-8
AROUNDED_DECIMALS = 5


def isclose(a: np.array, b: np.array) -> np.array:
    """
    numpy.iscloseの本モジュール用ラッパー。２つのnumpy.arrayの各要素を近似的に比較し、比較結果をnumpy.arrayで返す。

    Args:
        a (numpy.array): 比較対象1つ目
        b (numpy.array): 比較対象2つ目

    Return:
        numpy.array: 各要素の比較結果のnumpy.array (dtype=bool)
    """
    return np.isclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def allclose(a: np.array, b: np.array) -> bool:
    """
    numpy.allcloseの本モジュール用ラッパー。２つのnumpy.arrayの各要素を近似的に比較し、全て一致していたらTrueを返す。

    Args:
        a (numpy.array): 比較対象1つ目
        b (numpy.array): 比較対象2つ目

    Return:
        bool: 比較結果
    """
    return np.allclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def around(a: np.array) -> np.array:
    """
    numpy.aroundの本モジュール用ラッパー。numpy.arrayの各要素をモジュール指定の桁数で丸める

    Args:
        a (numpy.array): 比較対象1つ目

    Return:
        numpy.array: aを丸めた結果
    """
    return np.around(a, AROUNDED_DECIMALS)


def is_pow2(a: int) -> bool:
    """
    与えられた整数が2の累乗であるか判定する

    Args:
        a (int): 判定対象の整数

    Return:
        bool: 判定結果
    """
    # 2より小さい値の場合は偽
    if a < 2:
        return False

    # ビットの論理積で計算する
    return not a & (a - 1)


def is_probabilities(target_list: List[float]) -> bool:
    """
    浮動小数点数のリストが確率分布に対応するか判定する

    Args:
        target_list (List[float]): 判定対象の浮動小数点数リスト

    Return:
        bool: 判定結果
    """
    total = 0.0
    for element in target_list:
        if element < 0.0:
            return False
        total += element
        if round(total, AROUNDED_DECIMALS) > 1.0:
            return False

    if round(total, AROUNDED_DECIMALS) < 1.0:
        return False

    return True


def is_real(array: np.array) -> bool:
    """
    与えられたnumpy.arrayのデータ型が近似的に実数であるか判定する

    Args:
        array (numpy.array): 判定対象のnumpy.array

    Return:
        bool: 判定結果
    """
    if np.any(around(np.imag(array)) != 0j):
        return False

    return True
