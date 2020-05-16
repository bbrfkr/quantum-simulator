"""
QPU状態を表現するクラス群
"""

from quantum_simulator.base.qubits import Qubits
from quantum_simulator.channel.registers import Registers


class State:
    """
    QPU内における各変換前後の状態を表すクラス

    Attributes:
        qubits (Qubits): QPU内のQubit系
        registers (Registers): QPU内の古典レジスタ群
    """
    def __init__(self, qubits: Qubits, registers: Registers):
        """
        Args:
            qubits (Qubits): QPU内のQubit系
            registers (Registers): QPU内の古典レジスタ群
        """
        self.qubits = qubits
        self.registers = registers
