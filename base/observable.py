from math import ceil
from random import choices

from base.conf import approx_digit
from base.error import InitializeError
from base.qubits import Qubits, inner, is_orthogonal


class ObserveBasis:
    """観測基底のクラス"""

    def __init__(self, qubits_group: [Qubits]):
        # 観測基底を構成するQubit群同士は互いに直交していなければならない
        len_qubits_group = len(qubits_group)
        for index_0 in range(ceil(len_qubits_group / 2)):
            for index_1 in range(len_qubits_group - index_0 - 1):
                if not is_orthogonal(
                    qubits_group[index_0], qubits_group[len_qubits_group - index_1 - 1]
                ):
                    raise InitializeError("[ERROR]: 観測基底が直交しません")

        # 観測基底を構成するqubit列を初期化
        self.qubits_group = qubits_group


# class Observable:
#     """観測量のクラス"""

#     def __init__(
#         self, observed_value_0: float, observed_value_1: float, basis: ObserveBasis
#     ):
#         # 観測量は状態を識別しなければならないため、同じ観測値を与えてはならない
#         if round(observed_value_0 - observed_value_1, approx_digit) == 0.0:
#             raise InitializeError("[ERROR]: この観測量は状態を識別することができません")

#         # 観測値と観測対象のQubit
#         self.observed_values = [
#             {"value": observed_value_0, "qubit": basis.qubits[0]},
#             {"value": observed_value_1, "qubit": basis.qubits[1]},
#         ]

#     def expected_value(self, target: Qubit) -> float:
#         """対象Qubitに対する観測量の期待値を返す"""
#         expected_value_0 = (
#             self.observed_values[0]["value"]
#             * abs(inner(self.observed_values[0]["qubit"], target)) ** 2
#         )
#         expected_value_1 = (
#             self.observed_values[1]["value"]
#             * abs(inner(self.observed_values[1]["qubit"], target)) ** 2
#         )
#         return expected_value_0 + expected_value_1

#     def observe(self, target: Qubit) -> float:
#         """観測を実施して観測値を取得し、Qubitを収束させる"""
#         observed_probabilities = [
#             abs(inner(self.observed_values[index]["qubit"], target)) ** 2
#             for index in [0, 1]
#         ]
#         observed_result = choices(self.observed_values, observed_probabilities)[0]

#         # 観測によるQubitの収束
#         target.amplitudes = observed_result["qubit"].amplitudes

#         # 観測値の返却
#         return observed_result["value"]
