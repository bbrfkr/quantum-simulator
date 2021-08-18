"""
チャネル通過時の初期化するクラス群
"""

import random

from quantum_simulator.base import qubits, time_evolution
from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.utils import count_bits, is_real_close
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
    確保したビットはすべて|0>もしくは0となる

    Attributes:
        qubit_count (int): 確保するQubit数
        register_count (int): 確保する古典レジスタ数
    """

    # 値のバリデーション
    def __init__(self, qubit_count: int, register_count: int):
        """
        Args:
            qubit_count (int): 確保するQubit数
            register_count (int): 確保する古典レジスタ数
        """
        if register_count < 0 or qubit_count < 0:
            message = "[ERROR]: レジスタ数またはQubit数として、負の値が与えられました"
            raise InitializeError(message)

        self.qubit_count = qubit_count
        self.register_count = register_count

    def allocate(self) -> State:
        """
        要求された量子ビットおよび古典レジスタを用意する

        Returns:
            State: 量子ビットと古典レジスタを用意した直後の状態
        """
        init_qubits = ZERO

        # 二番目以降のQubitの結合
        for index in range(self.qubit_count - 1):
            init_qubits = qubits.combine(ZERO, init_qubits)

        registers = Registers(self.register_count)
        for index in range(self.register_count):
            registers.put(index, 0.0)

        return State(init_qubits, registers)


class Initializer:
    """
    Allocatorが確保した量子ビットを初期化するクラス

    Attributes:
        allocator (Allocator): Allocatorインスタンス
        initializer (List[Transformer]): 初期化の際に利用するTransformerの列
    """

    def __init__(self, allocator: Allocator, initializer=[]):
        """
        Args:
            allocator (Allocator): Allocatorインスタンス
            initializer (List[Transformer]): 初期化の際に利用するTransformerの列
        """
        self.allocator = allocator
        self.initializer = initializer

    def initialize(self) -> State:
        """
        量子ビットと古典レジスタ確保後、Transformer群による変換をかけて、量子ビットを初期化する
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
            target_count = count_bits(noise.matrix.shape[0]) - 1
            if qubit_count != target_count:
                message = "[ERROR]: 時間発展の対象Qubit系と用意されたQubit系は対応しません"
                raise InitializeError(message)

            noise_transformer = TimeEvolveTransformer(noise)
            mid_state = noise_transformer.transform(mid_state)

        # レジスタとインプットを比較して、初期状態を作るための時間発展を構成する
        input = self.allocator.input
        registers = mid_state.registers

        init_evolution = None
        if is_real_close((input & 0b1), registers.get(0)):
            init_evolution = IDENT_EVOLUTION
        else:
            init_evolution = NOT_GATE

        for index in range(qubit_count - 1):
            if is_real_close(((input >> index + 1) & 0b1), registers.get(index + 1)):
                init_evolution = time_evolution.combine(IDENT_EVOLUTION, init_evolution)
            else:
                init_evolution = time_evolution.combine(NOT_GATE, init_evolution)

        # 初期状態の作成
        init_transformer = TimeEvolveTransformer(init_evolution)
        init_state = init_transformer.transform(mid_state)

        return init_state
