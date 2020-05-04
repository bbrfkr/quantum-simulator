"""
純粋状態のQubit系の定義
"""

from math import ceil, sqrt
from typing import List, Tuple

import numpy as np
from numpy import conjugate
from numpy.linalg import linalg as LA

from quantum_simulator.base.conf import APPROX_DIGIT
from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    QubitCountNotMatchError,
)


class PureQubits:
    """
    一般的に複数かつ純粋状態のQubit群
      array: ndarray形式のQubits

    """

    def __init__(self, amplitudes: list):
        """
        初期化
          amplitudes: 確率振幅のリスト。ベクトル形式とndarray形式を許容する
        """

        # Qubit系であるかチェック
        tmp_array = np.array(amplitudes, dtype=complex)
        if not is_pure_qubits(tmp_array):
            message = "[ERROR]: 与えられたリストはQubit系に対応しません"
            raise InitializeError(message)

        # 各Qubit表現形式の導出
        vector, array = resolve_arrays(tmp_array)

        # 内包するQubit数を計算
        qubit_count = count_qubits(array)

        ### ここまでリファクタ済み 2020/05/04







        # 射影作用素を導出
        projection = np.multiply.outer(array, np.conjugate(array))

        # 射影作用素に対応する行列を求める
        projection_matrix_dim = 2 ** qubit_count
        projection_matrix = projection.reshape(
            projection_matrix_dim, projection_matrix_dim
        )

        # 初期化
        self.array = array
        self.qubit_count = qubit_count
        self.vector = vector
        self.projection = projection
        self.projection_matrix = projection_matrix
        self.projection_matrix_dim = projection_matrix_dim

    def __str__(self):
        """PureQubitsのベクトル表現を出力"""
        return str(self.vector)

    def print_array(self):
        """PureQubitsのndarray表現を出力"""
        print(self.array)

    def print_projection_matrix(self):
        """PureQubitsの射影行列を出力"""
        print(self.projection_matrix)

    def print_projection(self):
        """PureQubitsの射影行列に対するndarray表現を出力"""
        print(self.projection)

    def dirac_notation(self):
        """PureQubitsのDirac表記を出力"""
        term = ""
        array_repl = list(self.array.flat)
        for index in range(self.array.size):
            vec_repl = format(index, "b").zfill(len(self.array.shape))
            term += f"{array_repl[index]}|{vec_repl}>"

            # 最後以外はプラスと改行をつける
            if index != self.array.size - 1:
                term += " +\n"

        print(term)


def is_pure_qubits(array: np.array) -> bool:
    """与えられたarrayがQubit系を表現しているか判定する"""

    # 要素数が2の階乗個であるかチェック
    size = array.size
    if size == 0:
        return False

    while True:
        if size % 2 != 0:
            return False
        size /= 2
        if size % 2 == 1:
            break

    # ndarray形式の場合は、shapeの構成要素が全て2であるか
    # つまり、次元がQubitのテンソル積空間の次元かをチェック
    if len(array.shape) > 1:
        for sub_dim in array.shape:
            if sub_dim != 2:
                return False

    # 長さが1、つまり確率が1になるかをチェック
    norm = np.sqrt(np.sum(np.abs(array) ** 2))
    if np.round(norm - 1.0, APPROX_DIGIT) != 0.0:
        return False

    return True


def count_qubits(pure_qubits: np.array) -> int:
    """
    与えられたarrayがQubit系であることを仮定し
    Qubitの個数を返す
    """
    size = pure_qubits.size
    count = 0
    while True:
        size /= 2
        count += 1
        if size % 2 == 1:
            break
    return count


def resolve_arrays(pure_qubits: np.array) -> Tuple[np.array, np.array]:
    """
    与えられたarrayがQubit系であることを仮定し、
    ベクトル表現とndarray表現の組を返す
    """
    vector = None
    array = None

    if len(pure_qubits.shape) == 1:
        vector = pure_qubits
        qubit_count = count_qubits(pure_qubits)
        array_shape = tuple([2 for i in range(qubit_count)])
        array = pure_qubits.reshape(array_shape)
    else:
        array = pure_qubits
        vector = pure_qubits.reshape(pure_qubits.size)

    return (vector, array)


def combine(qubits_0: PureQubits, qubits_1: PureQubits) -> PureQubits:
    """二つのPureQubitsを結合する"""
    new_array = np.tensordot(qubits_0.array, qubits_1.array, 0)
    new_qubits = PureQubits(new_array)

    return new_qubits


def inner(qubits_0: PureQubits, qubits_1: PureQubits) -> complex:
    """PureQubit同士の内積 <qubit_0 | qubit_1>"""
    # 内積をとるQubit群同士のQubit数が一致してなければエラー
    if qubits_0.qubit_count != qubits_1.qubit_count:
        message = "[ERROR]: 対象PureQubits同士のQubit数が一致しません"
        raise QubitCountNotMatchError(message)

    return np.inner(qubits_0.array.flat, conjugate(qubits_1.array.flat))


def is_orthogonal(qubits_0: PureQubits, qubits_1: PureQubits) -> bool:
    """二つのPureQubits同士が直交しているか"""
    return np.round(inner(qubits_0, qubits_1), APPROX_DIGIT) == 0.0


def is_all_orthogonal(qubits_group: List[PureQubits]) -> bool:
    """複数のPureQubits同士が互い直交しているか"""
    len_pure_qubits_group = len(qubits_group)
    # PureQubitsが一つも入力されない時はエラー
    if len_pure_qubits_group == 0:
        message = "[ERROR]: 与えられたリストにPureQubitsが見つかりません"
        raise NoQubitsInputError(message)

    # PureQubitsが一つだけ与えられた時は明らかに互いに直交
    if len_pure_qubits_group == 1:
        return True

    # PureQubitsが二つ以上与えられた場合
    for index_0 in range(ceil(len_pure_qubits_group / 2)):
        for index_1 in range(len_pure_qubits_group - index_0 - 1):
            if not is_orthogonal(
                qubits_group[index_0], qubits_group[len_pure_qubits_group - index_1 - 1]
            ):
                return False

    return True
