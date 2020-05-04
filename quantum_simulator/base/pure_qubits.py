"""
純粋状態のQubit系の定義
"""

from math import ceil
from typing import List

import numpy as np
from numpy import conjugate
from numpy.linalg import linalg as LA

from .conf import APPROX_DIGIT
from .error import InitializeError, NoQubitsInputError, QubitCountNotMatchError


class PureQubits:
    """純粋状態の一般的に複数のQubit"""

    def __init__(self, amplitudes: list):
        array = np.array(amplitudes, dtype=complex)

        # 与えられた確率振幅の次元がQubitのテンソル積空間の次元かチェック
        for dim in array.shape:
            if dim != 2:
                message = "[ERROR]: 与えられた確率振幅の次元が不正です"
                raise InitializeError(message)

        # 確率の総和をチェック
        if np.round(LA.norm(array) - 1.0, APPROX_DIGIT) != 0.0:
            message = "[ERROR]: 確率の総和が1ではありません"
            raise InitializeError(message)

        # 内包するQubit数を計算
        qubit_count = len(array.shape)

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
        self.vector = self.array.reshape(self.array.size)
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
