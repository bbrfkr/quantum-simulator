"""
チャネルを構成するQPU状態変換のクラス群
"""

from abc import ABC, abstractmethod

from quantum_simulator.base.observable import Observable, observe
from quantum_simulator.base.time_evolution import TimeEvolution
from quantum_simulator.channel.state import State


class Transformer(ABC):
    """
    QPU状態変換の抽象クラス
    """

    @abstractmethod
    def transform(self, state: State, index=None) -> State:
        """
        QPU状態変換の抽象メソッド

        Args:
            state (State): 変換前のQPU状態
            index (Optional[int]): 古典情報を格納するレジスタ番号

        Returns:
            State: 変換後のQPU状態
        """
        pass


class ObserveTransformer(Transformer):
    """
    観測によるQPU状態変換のクラス
    """

    def __init__(self, observable: Observable):
        """
        Args:
            state (State): 変換前のQPU状態
            observable (Observable): 変換に利用する観測量
        """
        self.observable = observable

    def transform(self, state: State, index=None) -> State:
        """
        QPU状態変換のメソッド

        Args:
            state (State): 変換前のQPU状態
            index (Optional[int]): 古典情報を格納するレジスタ番号。指定されなければ観測結果は捨てられます

        Returns:
            State: 変換後のQPU状態
        """
        observed_value, converged_qubits = observe(self.observable, state.qubits)
        new_registers = state.registers
        if index is not None:
            new_registers.put(index, observed_value)

        return State(converged_qubits, new_registers)


class TimeEvolveTransformer(Transformer):
    """
    時間発展によるQPU状態変換のクラス
    """

    def __init__(self, time_evolution: TimeEvolution):
        """
        Args:
            time_evolution (TimeEvolution): 変換に利用する時間発展
        """
        self.time_evolution = time_evolution

    def transform(self, state: State, index=None) -> State:
        """
        QPU状態変換のメソッド
        
        Args:
            state (State): 変換前のQPU状態
            index (Optional[int]): 古典情報を格納するレジスタ番号。本変換では無視されます

        Returns:
            State: 変換後のQPU状態
        """
        transformed_qubits = self.time_evolution.operate(state.qubits)

        return State(transformed_qubits, state.registers)
