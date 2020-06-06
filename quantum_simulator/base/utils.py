"""
baseパッケージ内で利用するユーティリティメソッド群
"""

from typing import List

import numpy
import math

from quantum_simulator.base.error import NegativeValueError
from quantum_simulator.base.switch_cupy import xp_factory

# 計算時の近似桁数
RELATIVE_TOLERANCE = 1.0e-5
ABSOLUTE_TOLERANCE = 1.0e-8
AROUNDED_DECIMALS = 5

np = xp_factory()  # typing: numpy


def allclose(a: numpy.array, b: numpy.array) -> bool:
    """
    numpy.allcloseの本モジュール用ラッパー。２つのnp.arrayの各要素を近似的に比較し、全て一致していたらTrueを返す。

    Args:
        a (np.array): 比較対象1つ目
        b (np.array): 比較対象2つ目

    Return:
        bool: 比較結果
    """
    return np.all(around(a - b) == 0.0 + 0j)
    
    #np.allclose(a, b, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def isclose(a: complex, b: complex) -> bool:
    """
    2つの複素数を近似的に比較し、一致していたらTrueを返す。

    Args:
        a (complex): 比較対象1つ目
        b (complex): 比較対象2つ目

    Return:
        bool: 比較結果
    """
    return math.isclose(a.real, b.real, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE) and math.isclose(a.imag, b.imag, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


def around(a: numpy.array) -> numpy.array:
    """
    numpy.aroundの本モジュール用ラッパー。np.arrayの各要素をモジュール指定の桁数で丸める

    Args:
        a (np.array): 比較対象1つ目

    Return:
        np.array: aを丸めた結果
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


def count_bits(a: int) -> int:
    """
    与えられた非負整数が表現可能な最低ビット数を返す

    Args:
        a (int): 判定対象の整数

    Return:
        int: 表現可能な最低ビット数
    """
    # 負の整数が与えられた時はエラー
    if a < 0:
        message = "[ERROR]: 負数が与えられました"
        raise NegativeValueError(message)

    bit_count = 0
    while True:
        a //= 2
        bit_count += 1
        if a == 0:
            break

    return bit_count


def is_probabilities(target_list: List[float]) -> bool:
    """
    浮動小数点数のリストが確率分布に対応するか判定する

    Args:
        target_list (List[float]): 判定対象の浮動小数点数リスト

    Return:
        bool: 判定結果
    """
    target_array = np.array(target_list)

    if np.any(around(target_array) < 0.0):
        return False

    if not math.isclose(np.sum(target_array), 1.0):
        return False

    return True


def is_real(array: numpy.array) -> bool:
    """
    与えられたnp.arrayのデータ型が近似的に実数であるか判定する

    Args:
        array (np.array): 判定対象のnp.array

    Return:
        bool: 判定結果
    """
    if np.any(around(np.imag(array)) != 0j):
        return False

    return True
