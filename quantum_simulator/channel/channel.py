"""
量子チャネルを表現するクラス群
"""

from abc import ABC, abstractmethod
from typing import List

from quantum_simulator.base.time_evolution import TimeEvolution
from quantum_simulator.channel.finalizer import Finalizer
from quantum_simulator.channel.initializer import Allocator, Initializer
from quantum_simulator.channel.state import State
from quantum_simulator.channel.transformer import Transformer


class Channel(ABC):
    """
    量子チャネルの抽象クラス

    Attributes:
        qubit_count (int): チャネル内のqubit数
        register_count (int): チャネル内の古典レジスタ数
        noise (Optional[TImeEvolution]): 初期化時に発生し得るノイズとしての時間発展
        output_indices (List[int]): 出力結果を観測するQubit番号のリスト
        transformers (List[Transformer]): 変換の列
        states (List(State)): QPU状態の列
    """

    def __init__(self, qubit_count: int, register_count: int, noise=None):
        """
        Args:
            qubit_count (int): チャネル内のqubit数
            register_count (int): チャネル内の古典レジスタ数
            noise (Optional[TImeEvolution]): 初期化時に発生し得るノイズとしての時間発展
        """
        self.qubit_count = qubit_count
        self.register_count = register_count
        self.noise = noise  # type: Optional[TimeEvolution]
        self.output_indices = output_indices
        self.transformers = []  # type: List[Transformer]
        self.states = []  # type: List[State]

    def initialize(self, input: int) -> void:
        """
        Transformer配列とState配列を初期化する

        Args:
            input (int): 入力情報
        """
        allocator = Allocator(self.input, self.qubit_count, self.register_count)
        initializer = Initializer(allocator, self.noise)
        self.transformers = []
        self.states = [initializer.initialize()]

    def transform(self, transformer: Transformer, index=None) -> void:
        """
        Transformerを用い、状態を次の状態に遷移させる

        Args:
            transformer (Transformer): 利用する状態変換
            index (Optionnal[int]): 古典情報が得られた場合に格納するレジスタ番号
        """
        self.transformers.append(transformer)
        self.states.append(transformer.transform(states[-1], index))

    def finalize(self, output_indices: List[int]) -> int:
        """
        最終処理を行い、計算結果を取得する

        Args:
            output_indices (List[int]): 出力結果を観測するQubit番号のリスト

        Returns:
            int: 最終的な計算結果
        """
        finalizer = Finalizer(output_indices)
        return finalizer.finalize()

    def apply(self, input: int, output_indices: List[int]) -> int:
        """
        CPUから見たQPUによる計算の適用
        """
        self.initialize(input)
        self.calculate()
        return self.finalize(output_indices)

    @abstractmethod
    def calculate(self) -> void:
        """
        ユーザ実装の量子チャネルの計算部
        """
        pass
