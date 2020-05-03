"""
純粋状態のQubit系の定義
"""

from math import ceil
from typing import List

import numpy as np
from numpy import conjugate
from numpy.linalg import linalg

from .conf import APPROX_DIGIT
from .error import InitializeError, NoQubitsInputError, QubitCountNotMatchError


class PureQubits:
    """純粋状態の一般的に複数のQubit"""

    def __init__(self, amps: list):
        amplitudes = np.array(amps, dtype=complex)

        # 与えられた確率振幅の次元がQubitのテンソル積空間の次元かチェック
        for dim in amplitudes.shape:
            if dim != 2:
                message = "[ERROR]: 与えられた確率振幅の次元が不正です"
                raise InitializeError(message)

        # 確率の総和をチェック
        if np.round(linalg.norm(amplitudes) - 1.0, APPROX_DIGIT) != 0.0:
            message = "[ERROR]: 確率の総和が1ではありません"
            raise InitializeError(message)

        # 内包するQubit数を計算
        qubit_count = len(amplitudes.shape)

        # 射影作用素を導出
        projection = np.multiply.outer(amplitudes, np.conjugate(amplitudes))

        # 射影作用素に対応する行列を求める
        matrix_dim = 2 ** qubit_count
        matrix = projection.reshape(matrix_dim, matrix_dim)

        # 初期化
        self.amplitudes = amplitudes
        self.qubit_count = qubit_count
        self.vector = self.amplitudes.reshape(self.amplitudes.size)
        self.projection = projection
        self.matrix = matrix
        self.matrix_dim = matrix_dim

    def __str__(self):
        """PureQubitsのベクトル表現を出力"""
        return str(self.vector)

    def print_array(self):
        """PureQubitsのndarray表現を出力"""
        print(self.amplitudes)

    def dirac_notation(self):
        """PureQubitsのDirac表記を出力"""
        term = ""
        array_repl = list(self.amplitudes.flat)
        for index in range(self.amplitudes.size):
            vec_repl = format(index, "b").zfill(len(self.amplitudes.shape))
            term += f"{array_repl[index]}|{vec_repl}>"

            # 最後以外はプラスと改行をつける
            if index != self.amplitudes.size - 1:
                term += " +\n"

        print(term)


def combine(qubits_0: PureQubits, qubits_1: PureQubits) -> PureQubits:
    """二つのPureQubitsを結合する"""
    new_amplitudes = np.tensordot(qubits_0.amplitudes, qubits_1.amplitudes, 0)
    new_qubits = PureQubits(new_amplitudes)

    return new_qubits


def inner(qubits_0: PureQubits, qubits_1: PureQubits) -> complex:
    """PureQubit同士の内積 <qubit_0 | qubit_1>"""
    # 内積をとるQubit群同士のQubit数が一致してなければエラー
    if qubits_0.qubit_count != qubits_1.qubit_count:
        message = "[ERROR]: 対象PureQubits同士のQubit数が一致しません"
        raise QubitCountNotMatchError(message)

    return np.inner(qubits_0.amplitudes.flat, conjugate(qubits_1.amplitudes.flat))


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


# def reduction(density: np.ndarray, target: int) -> np.ndarray:
#     """target番目のQubitを縮約した、局所Qubit群に対応する密度行列を返す"""
#     qubit_count = int(len(density.shape) / 2)
#     if qubit_count == 1:
#         message = "[ERROR]: このQubit系はこれ以上縮約できません"
#         raise ReductionError(message)
#     return np.trace(
#         density, axis1=qubit_count - 1 - target, axis2=(2 * qubit_count - 1 - target),
#     )
