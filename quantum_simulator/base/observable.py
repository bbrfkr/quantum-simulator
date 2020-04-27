from random import choices

from base.conf import approx_digit
from base.error import CannotDistinguishError, NonOrthogonalError
from base.qubit import Qubit, inner, is_orthogonal


class ObserveBasis:
    """観測基底のクラス"""

    def __init__(self, qubit_0: Qubit, qubit_1: Qubit):
        # 観測基底を構成するQubit同士は直交していなければならない
        if not is_orthogonal(qubit_0, qubit_1):
            raise NonOrthogonalError("[ERROR]: 観測基底が直交しません")

        # 観測基底を構成するqubit列を初期化
        self.qubits = []
        self.qubits.append(qubit_0)
        self.qubits.append(qubit_1)


class Observable:
    """観測量のクラス"""

    def __init__(
        self, observed_value_0: float, observed_value_1: float, basis: ObserveBasis
    ):
        # 観測量は状態を識別しなければならないため、同じ観測値を与えてはならない
        if round(observed_value_0 - observed_value_1, approx_digit) == 0.0:
            raise CannotDistinguishError("[ERROR]: この観測量は状態を識別することができません")

        # 観測値と観測対象のQubit
        self.observed_values = [
            {"value": observed_value_0, "qubit": basis.qubits[0]},
            {"value": observed_value_1, "qubit": basis.qubits[1]},
        ]

    def expected_value(self, target: Qubit) -> float:
        """対象Qubitに対する観測量の期待値を返す"""
        expected_value_0 = (
            self.observed_values[0]["value"]
            * abs(inner(self.observed_values[0]["qubit"], target)) ** 2
        )
        expected_value_1 = (
            self.observed_values[1]["value"]
            * abs(inner(self.observed_values[1]["qubit"], target)) ** 2
        )
        return expected_value_0 + expected_value_1

    def observe(self, target: Qubit) -> float:
        """観測を実施して観測値を取得し、Qubitを収束させる"""
        observed_probabilities = [
            abs(inner(self.observed_values[index]["qubit"], target)) ** 2
            for index in [0, 1]
        ]
        observed_result = choices(self.observed_values, observed_probabilities)[0]

        # 観測によるQubitの収束
        target.amplitudes = observed_result["qubit"].amplitudes

        # 観測値の返却
        return observed_result["value"]
