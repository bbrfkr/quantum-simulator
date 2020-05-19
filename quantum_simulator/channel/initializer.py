"""
チャネル通過時の初期化するクラス群
"""

import random

from quantum_simulator.base import qubits, time_evolution
from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.utils import allclose, count_bits
from quantum_simulator.channel.registers import Registers
from quantum_simulator.channel.state import State
from quantum_simulator.channel.transformer import TimeEvolveTransformer
from quantum_simulator.major.qubits import ONE, ZERO
from quantum_simulator.major.time_evolution import (
    IDENT_EVOLUTION,
    NOT_GATE,
    TimeEvolution,
)


class Allocator:
    """
    必要なだけ量子ビットと古典レジスタを用意するクラス

    Args:
        input (int): 入力値(非負値)
        qubit_count (int): 確保するQubit数
        register_count (int): 確保する古典レジスタ数
    """

    # 値のバリデーション
    def __init__(self, input: int, qubit_count: int, register_count: int):
        """
        Args:
            input (int): 入力値(非負値)
            qubit_count (int): 確保するQubit数
            register_count (int): 確保する古典レジスタ数
        """
        if input < 0:
            message = "[ERROR]: 入力値として負の値が与えられました"
            raise InitializeError(message)

        bit_count = count_bits(input)
        if register_count < bit_count:
            message = "[ERROR]: レジスタ数が入力値を表現可能な最小ビット数を満たしません"
            raise InitializeError(message)

        if bit_count > qubit_count:
            message = "[ERROR]: 確保するQubit数が入力値を表現するには不足しています"
            raise InitializeError(message)

        self.input = input
        self.qubit_count = qubit_count
        self.register_count = register_count

    def allocate(self) -> State:
        """
        要求された量子ビットおよび古典レジスタを用意する

        Returns:
            State: 量子ビットと古典レジスタを用意した直後の状態
        """
        # 初期Qubitの用意
        init_qubits = None
        registers = Registers(self.register_count)
        candidates_list = [0, 1]
        selected = random.choice(candidates_list)
        if selected == 0:
            init_qubits = ZERO
            registers.put(0, 0.0)

        else:
            init_qubits = ONE
            registers.put(0, 1.0)

        # 二番目以降のQubitの結合
        registers = Registers(self.register_count)
        for index in range(self.qubit_count - 1):
            selected = random.choice(candidates_list)
            if selected == 0:
                init_qubits = qubits.combine(init_qubits, ZERO)
                registers.put(index + 1, 0.0)
            else:
                init_qubits = qubits.combine(init_qubits, ONE)
                registers.put(index + 1, 1.0)

        # インスタンスの初期値の代入
        return State(init_qubits, registers)


class Initializer:
    """
    Allocatorが確保した量子ビットを初期化するクラス

    Args:
        allocator (Allocator): Allocatorインスタンス
        noise (Optional[TimeEvolution]): ノイズとして作用する任意の時間発展
    """

    def __init__(self, allocator: Allocator, noise=None):
        """
        Args:
            allocator (Allocator): Allocatorインスタンス
            noise (Optional[TimeEvolution]): ノイズとして作用する任意の時間発展
        """
        self.allocator = allocator
        self.noise = noise

    def initialize(self) -> State:
        """
        量子ビットと古典レジスタ確保後、ノイズをかけたのち、状態初期化の時間発展を適用する
        最終的に初期化された状態(QubitsとRegistersの組)を返す

        Returns:
            State: 最終的な初期化状態
        """

        mid_state = self.allocator.allocate()
        qubit_count = mid_state.qubits.qubit_count

        if self.noise is not None and isinstance(self.noise, TimeEvolution):
            # 時間発展の対象Qubit数とAllocatorの保持しているQubit数が
            # 一致しなかったら、エラー
            noise = self.noise  # type: TimeEvolution
            target_count = len(noise.ndarray.shape)
            if qubit_count != target_count:
                message = "[ERROR]: 時間発展の対象Qubit系と用意されたQubit系は対応しません"
                raise InitializeError(message)

            noise_transformer = TimeEvolveTransformer(mid_state, noise)
            mid_state = noise_transformer.transform()

        # レジスタとインプットを比較して、初期状態を作るための時間発展を構成する
        input = self.allocator.input
        bit_count = count_bits(self.allocator.input)
        registers = mid_state.registers

        init_evolution = None
        if allclose((input & 0b1), registers.get(0)):
            init_evolution = IDENT_EVOLUTION
        else:
            init_evolution = NOT_GATE

        for index in range(bit_count - 1):
            if allclose(((input & 0b1) >> index + 1), registers.get(index + 1)):
                init_evolution = time_evolution.combine(init_evolution, IDENT_EVOLUTION)
            else:
                init_evolution = time_evolution.combine(init_evolution, NOT_GATE)

        for index in range(qubit_count - bit_count):
            init_evolution = time_evolution.combine(init_evolution, IDENT_EVOLUTION)

        # 初期状態の作成
        init_transformer = TimeEvolveTransformer(mid_state, init_evolution)
        init_state = init_transformer.transform()

        return init_state
