from numpy import conjugate
from numpy import inner as np_inner
from numpy.linalg import linalg

from base.conf import approx_digit
from base.error import InitializeError


class Qubit:
    """
    純粋状態のqubit (i.e., amp_0 |0> + amp_1 |1>)
    """

    def __init__(self, amp_0: complex, amp_1: complex):
        # 近似桁数で丸めても、qubitのノルムが1にならないならエラー
        if round(linalg.norm([amp_0, amp_1]) - 1, approx_digit) != 0.0:
            raise InitializeError("[ERROR]: Qubitの長さが不正です")

        # Qubitを初期化
        self.amplitudes = []
        self.amplitudes.append(amp_0)
        self.amplitudes.append(amp_1)

    def __str__(self):
        return f"{self.amplitudes[0]}|0> + {self.amplitudes[1]}|1>"


def inner(qubit_0: Qubit, qubit_1: Qubit):
    """Qubit同士の内積 <qubit_0 | qubit_1>"""
    return np_inner(qubit_0.amplitudes, conjugate(qubit_1.amplitudes))


def is_orthogonal(qubit_0: Qubit, qubit_1: Qubit):
    """Qubit同士が直交しているか"""
    return round(inner(qubit_0, qubit_1), approx_digit) == 0.0
