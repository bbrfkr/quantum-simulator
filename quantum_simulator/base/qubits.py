from math import ceil

import numpy as np
from numpy import conjugate
from numpy.linalg import linalg

from .conf import approx_digit
from .error import InitializeError, NoQubitsInputError, QubitCountNotMatchError


class Qubits:
    """純粋状態の一般的に複数のQubit"""

    def __init__(self, amplitudes: np.array):
        # 要素が複素数かをチェック
        if amplitudes.dtype != complex:
            message = "[ERROR]: 与えられた確率振幅のデータタイプが複素数ではありません"
            raise InitializeError(message)

        # 与えられた確率振幅の次元がQubitのテンソル積空間の次元かチェック
        for dim in amplitudes.shape:
            if dim != 2:
                message = "[ERROR]: 与えられた確率振幅の次元が不正です"
                raise InitializeError(message)

        # 確率の総和をチェック
        if round(linalg.norm(amplitudes) - 1.0, approx_digit) != 0.0:
            message = "[ERROR]: 確率の総和が1ではありません"
            raise InitializeError(message)

        # 初期化
        self.amplitudes = np.array(amplitudes, complex)
        self.qubit_count = len(self.amplitudes.shape)

    def __str__(self):
        term = ""
        array_repl = list(self.amplitudes.flat)
        for index in range(self.amplitudes.size):
            vec_repl = format(index, "b").zfill(len(self.amplitudes.shape))
            term += f"{array_repl[index]}|{vec_repl}>"

            # 最後以外はプラスをつけ、その後改行
            if index != self.amplitudes.size - 1:
                term += " +"
            term += "\n"

        return term


def combine(qubits_0: Qubits, qubits_1: Qubits) -> Qubits:
    """二つのQubit群を結合する"""
    new_amplitudes = np.tensordot(qubits_0.amplitudes, qubits_1.amplitudes, 0)
    new_qubits = Qubits(new_amplitudes)

    # 元々あったリソースは削除して利用不能にする(コピー不可能性定理)
    del qubits_0.amplitudes, qubits_1.amplitudes

    return new_qubits


def inner(qubits_0: Qubits, qubits_1: Qubits) -> complex:
    """Qubit群同士の内積 <qubit_0 | qubit_1>"""
    # 内積をとるQubit群同士のQubit数が一致してなければエラー
    if qubits_0.qubit_count != qubits_1.qubit_count:
        message = "[ERROR]: 対象Qubit群同士のQubit数が一致しません"
        raise QubitCountNotMatchError(message)

    return np.inner(qubits_0.amplitudes.flat, conjugate(qubits_1.amplitudes.flat))


def is_orthogonal(qubits_0: Qubits, qubits_1: Qubits) -> bool:
    """二つのQubit群同士が直交しているか"""
    return round(inner(qubits_0, qubits_1), approx_digit) == 0.0


def is_all_orthogonal(qubits_group: [Qubits]) -> bool:
    """Qubit群同士が互い直交しているか"""
    len_qubits_group = len(qubits_group)
    # Qubit群が一つも入力されない時はエラー
    if len_qubits_group == 0:
        message = "[ERROR]: 与えられたリストにQubit群が見つかりません"
        raise NoQubitsInputError(message)

    # Qubit群が一つだけ与えられた時は明らかに互いに直交
    if len_qubits_group == 1:
        return True

    # Qubit群が二つ以上与えられた場合
    for index_0 in range(ceil(len_qubits_group / 2)):
        for index_1 in range(len_qubits_group - index_0 - 1):
            if not is_orthogonal(
                qubits_group[index_0], qubits_group[len_qubits_group - index_1 - 1]
            ):
                return False
            return True
