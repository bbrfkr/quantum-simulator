from typing import List, TypedDict

from quantum_simulator.base.qubits import Qubits


class ObservableElement(TypedDict):
    """観測量の構成要素を表すタイプ"""

    value: float
    qubits: Qubits


"""観測量の構成要素のリストを表すタイプ"""
ObservableElements = List[ObservableElement]
