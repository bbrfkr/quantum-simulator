from base.error import NonOrthogonalError
from base.qubit import Qubit, is_orthogonal


class ObserveBasis:
    """観測基底のクラス"""

    def __init__(self, qubit_0: Qubit, qubit_1: Qubit):
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
        # 観測値と観測対象のQubit
        self.observed_values = [
            {"value": observed_value_0, "qubit": basis.qubits[0]},
            {"value": observed_value_1, "qubit": basis.qubits[1]},
        ]

    def expected_value(self, target: Qubit):
        """対象Qubitに対する観測量の期待値を返す"""
        return None

    def observe(self, target: Qubit):
        """観測を実施して観測値を取得し、Qubitを収束させる"""
        return None
