"""
チャネルを構成するQPU状態変換のクラス
"""

from abc import ABC, abstractmethod
from typing import Optional

from quantum_simulator.base.observable import Observable, observe
from quantum_simulator.base.time_evolution import TimeEvolution
from quantum_simulator.channel.state import State


class Transformer(ABC):
    """
    QPU状態変換の抽象クラス
    """

    def __init__(self, state: State):
        """
        Args:
            state (State): 変換前のQPU状態
        """
        self.state = state

    @abstractmethod
    def transform(self, index: Optional[int]) -> State:
        """
        QPU状態変換の抽象メソッド
            index (Optional[int]): 古典情報を格納するレジスタ番号

        Returns:
            State: 変換後のQPU状態
        """
        pass


class ObserveTransformer(Transformer):
    """
    観測によるQPU状態変換のクラス
    """

    def __init__(self, state: State, observable: Observable):
        """
        Args:
            state (State): 変換前のQPU状態
            observable (Observable): 変換に利用する観測量
        """
        super().__init__(state)
        self.observable = observable

    def transform(self, index: Optional[int]) -> State:
        """
        QPU状態変換のメソッド
            index (Optional[int]): 古典情報を格納するレジスタ番号。指定されなければ観測結果は捨てられます

        Returns:
            State: 変換後のQPU状態
        """
        observed_value, converged_qubits = observe(self.observable, self.state.qubits)
        new_registers = self.state.registers
        if index is not None:
            new_registers.put(index, observed_value)

        return State(converged_qubits, new_registers)


class TimeEvolveTransformer(Transformer):
    """
    時間発展によるQPU状態変換のクラス
    """

    def __init__(self, state: State, time_evolution: TimeEvolution):
        """
        Args:
            state (State): 変換前のQPU状態
            time_evolution (TimeEvolution): 変換に利用する時間発展
        """
        super().__init__(state)
        self.time_evolution = time_evolution

    def transform(self, index: Optional[int]) -> State:
        """
        QPU状態変換のメソッド
            index (Optional[int]): 古典情報を格納するレジスタ番号。本変換では無視されます

        Returns:
            State: 変換後のQPU状態
        """
        transformed_qubits = self.time_evolution.operate(self.state.qubits)

        return State(transformed_qubits, self.state.registers)
