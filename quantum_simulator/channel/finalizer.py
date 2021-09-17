"""
終了処理を表現するクラス群
"""

from typing import List, Tuple

import numpy as np

from quantum_simulator.base.error import OutOfRangeIndexError
from quantum_simulator.base.observable import Observable, observe
from quantum_simulator.base.utils import around
from quantum_simulator.channel.state import State


class Finalizer:
    """
    終了処理を表すクラス

    Attributes:
        output_indices (List[int]): 観測対象のQubit番号の昇順リスト
    """

    def __init__(self, output_indices: List[int]):
        """
        Args:
            output_indices (List[int]): 観測対象のQubit番号のリスト
        """
        self.output_indices = sorted(output_indices)

    def finalize(self, state: State) -> Tuple[int, State]:
        """
        最終状態を観測し、計算結果を出力する

        Args:
            state (State): 観測対象の状態

        Returns:
            Tuple[int, State]: 最終的な計算結果と、収束後の状態
        """
        # Qubit番号のバリデーション
        qubit_count = state.qubits.qubit_count
        for index in self.output_indices:
            if index > qubit_count - 1 or index < 0:
                message = "観測対象のQubit番号に不正な値が含まれています"
                raise OutOfRangeIndexError(message)

        # 全系観測量の生成
        diagonal_values = [value for value in range(2 ** qubit_count)]
        observable = Observable(np.diag(diagonal_values))

        # 計算結果の観測とターゲットビット抽出
        raw_outcome, converged_qubits = observe(observable, state.qubits)
        int_outcome = around(np.array(raw_outcome)).astype(int)
        post_state = State(converged_qubits, state.registers)

        outcome = 0
        loop_index = 0
        for output_index in self.output_indices:
            target_bit = (int_outcome >> output_index) & 0b1
            outcome += target_bit * 2 ** loop_index
            loop_index += 1

        return (outcome, post_state)
