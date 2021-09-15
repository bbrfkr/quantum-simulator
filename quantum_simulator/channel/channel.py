"""
量子チャネルを表現するクラス群
"""

from typing import List

from quantum_simulator.channel.finalizer import Finalizer
from quantum_simulator.channel.initializer import Allocator, Initializer
from quantum_simulator.channel.state import State
from quantum_simulator.channel.transformer import Transformer


class Channel:
    """
    量子チャネルの抽象クラス

    Attributes:
        qubit_count (int): チャネル内のqubit数
        register_count (int): チャネル内の古典レジスタ数
        init_transformers (List[Transformer]): 初期状態を作成する変換の列
        transformers (List[Transformer]): シミュレーション目的の変換の列
        states (List(State)): QPU状態の列
    """

    def __init__(self, qubit_count: int, register_count: int, init_transformers=[]):
        """
        Args:
            qubit_count (int): チャネル内のqubit数
            register_count (int): チャネル内の古典レジスタ数
            init_transformers (List[Transformer]): 初期状態を作成する変換の列
        """
        self.qubit_count = qubit_count
        self.register_count = register_count
        self.init_transformers = init_transformers  # type: List[Transformer]
        self.transformers = []  # type: List[Transformer]
        self.states = []  # type: List[State]
        self.is_finalized = False

    def initialize(self):
        """
        Transformer配列とState配列を初期化する

        Args:
            input (int): 入力情報
        """
        allocator = Allocator(self.qubit_count, self.register_count)
        initializer = Initializer(allocator, self.init_transformers)
        self.transformers = []
        self.states = [initializer.initialize()]

    def transform(self, transformer: Transformer, index=None):
        """
        Transformerを用い、状態を次の状態に遷移させる

        Args:
            transformer (Transformer): 利用する状態変換
            index (Optionnal[int]): 古典情報が得られた場合に格納するレジスタ番号
        """
        self.transformers.append(transformer)
        self.states.append(transformer.transform(self.states[-1], index))

    def finalize(self, output_indices: List[int]) -> int:
        """
        最終処理を行い、計算結果を取得する

        Args:
            output_indices (List[int]): 出力結果を観測するQubit番号のリスト

        Returns:
            int: 最終的な計算結果
        """
        finalizer = Finalizer(output_indices)
        outcome = finalizer.finalize(self.states[-1])
        self.is_finalized = True
        return outcome
