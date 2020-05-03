from typing import List, TypedDict

from quantum_simulator.base.pure_qubits import PureQubits


class ObservableElement(TypedDict):
    """観測量の構成要素を表すタイプ"""

    value: float
    qubits: PureQubits


"""観測量の構成要素のリストを表すタイプ"""
ObservableElements = List[ObservableElement]
