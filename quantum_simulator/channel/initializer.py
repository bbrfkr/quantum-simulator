"""
チャネル通過時の初期化するクラス群
"""

from quantum_simulator.base import qubits
from quantum_simulator.base.error import InitializeError
from quantum_simulator.channel.registers import Registers
from quantum_simulator.channel.state import State
from quantum_simulator.channel.transformer import Transformer
from quantum_simulator.major.qubits import ZERO


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
            message = "レジスタ数またはQubit数として、負の値が与えられました"
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
        transformers (List[Transformer]): 初期化の際に利用するTransformerの列
    """

    def __init__(self, allocator: Allocator, transformers=[]):
        """
        Args:
            allocator (Allocator): Allocatorインスタンス
            initializer (List[Transformer]): 初期化の際に利用するTransformerの列
        """
        self.allocator = allocator
        self.transformers = transformers

    def initialize(self) -> State:
        """
        量子ビットと古典レジスタ確保後、Transformer群による変換をかけて、量子ビットを初期化する
        最終的に初期化された状態(QubitsとRegistersの組)を返す

        Returns:
            State: 最終的な初期化状態
        """

        init_state = self.allocator.allocate()

        for transformer in self.transformers:
            if not isinstance(transformer, Transformer):
                message = "与えられた要素はTransformerではありません"
                raise InitializeError(message)

            init_state = transformer.transform(init_state)

        return init_state
