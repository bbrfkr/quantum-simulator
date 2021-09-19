"""
時間発展を記述するクラス群
"""

from typing import List, Optional, cast

from .error import (
    IncompatibleDimensionError,
    InitializeError,
    NotCompleteError,
    NotMatchCountError,
)
from .pure_qubits import OrthogonalSystem
from .qubits import Qubits, is_qubits_dim
from .switch_cupy import xp_factory
from .utils import allclose

np = xp_factory()  # typing: numpy


class TimeEvolution:
    """
    ユニタリ変換による時間発展のクラス

    Attributes:
        ndarray: ndarray形式のユニタリ変換
        matrix: 行列形式のユニタリ変換
    """

    def __init__(self, unitary_matrix: list):
        """
        Args:
            unitary_array: ユニタリ変換の候補となるリスト。行列形式とndarray形式を許容する
        """
        matrix = np.array(unitary_matrix)

        # 次元のチェック
        if not is_qubits_dim(matrix):
            message = "与えられたリストはQubit系上の作用素ではありません"
            raise InitializeError(message)

        # ユニタリ性のチェック
        hermite_matrix = matrix @ np.conj(matrix.T)
        if not allclose(hermite_matrix, np.identity(matrix.shape[0])):
            message = "与えられたリストはユニタリ変換ではありません"
            raise InitializeError(message)
        del hermite_matrix

        # 初期化
        self.matrix = matrix

    def __str__(self):
        """
        ユニタリ変換の行列表現の文字列を返す

        Returns:
            str: ユニタリ変換の行列表現の文字列
        """
        return str(self.matrix)

    def operate(self, qubits: Qubits) -> Qubits:
        """
        対象Qubitsを時間発展によって別のQubitsに変換し、変換後のQubitsを返す

        Args:
            qubits (Qubits): 変換対象のQubits

        Returns:
            Qubits: 変換後のQubits
        """
        if self.matrix.shape[0] != qubits.matrix.shape[0]:
            message = "変換対象のQubit数が不正です"
            raise IncompatibleDimensionError(message)

        transformed_matrix = self.matrix @ qubits.matrix @ np.conj(self.matrix.T)
        return Qubits(transformed_matrix)


def create_from_onb(
    pre_ons: OrthogonalSystem, post_ons: OrthogonalSystem
) -> TimeEvolution:
    """
    変換元基底と変換後基底を指定して、対応する時間発展を作る

    Args:
        pre_ons (OrthogonalSystem): 変換前の正規直交系。基底である必要がある
        post_ons (OrthogonalSystem): 変換後の正規直交系。基底である必要がある

    Returns:
        TimeEvolution: 導出された時間発展
    """
    # 指定されたONSが全てONBでなければエラー
    if not (pre_ons.is_onb() and post_ons.is_onb()):
        message = "正規直交基底を指定してください"
        raise NotCompleteError(message)

    # 観測基底を構成するQubit群の個数同士が一致していなければエラー
    len_pre_ons = len(pre_ons.qubits_list)
    if len_pre_ons != len(post_ons.qubits_list):
        message = "変換後のQubit数が変換前と異なります"
        raise InitializeError(message)

    # 変換のndarrayの生成
    elements_matrices = [
        np.outer(
            post_ons.qubits_list[index].vector,
            np.conj(pre_ons.qubits_list[index].vector),
        )
        for index in range(len_pre_ons)
    ]

    matrix = elements_matrices[-1]
    for index in range(len_pre_ons - 1):
        matrix = np.add(matrix, elements_matrices[index])

    del elements_matrices
    return TimeEvolution(matrix)


def combine(
    time_evolution_0: Optional[TimeEvolution], time_evolution_1: TimeEvolution
) -> TimeEvolution:
    """
    2つの時間発展を結合して合成系の時間発展を作る

    Args:
        time_evolution_0 (Optional[TimeEvolution]): 結合される側の時間発展
        time_evolution_1 (TimeEvolution): 結合する側の時間発展

    Returns:
        TimeEvolution: 結合後の時間発展
    """
    if time_evolution_0 is None:
        return time_evolution_1

    # 新しい時間発展の生成
    time_evolution_0_matrix = list(time_evolution_0.matrix)
    new_matrix = np.vstack(
        tuple(
            [
                np.hstack(
                    tuple(
                        [
                            element * time_evolution_1.matrix
                            for element in time_evolution_0_row
                        ]
                    )
                )
                for time_evolution_0_row in time_evolution_0_matrix
            ]
        )
    )
    return TimeEvolution(new_matrix)


def multiple_combine(evolutions: List[TimeEvolution]) -> TimeEvolution:
    """
    一般的に2つ以上のユニタリ変換を結合して合成系の時間発展を作る

    Args:
        evolutions (List[TimeEvolution]): 結合対象の時間発展のリスト

    Returns:
        TimeEvolution: 結合後の時間発展
    """
    if not evolutions:
        message = "空のリストが与えられました"
        raise NotMatchCountError(message)

    combined_evolution = None
    for evolution in evolutions:
        combined_evolution = combine(combined_evolution, evolution)

    # リストは空ではないかつ、combineは必ず値を返すことが保証されているのでキャストする
    casted_evolution = cast(TimeEvolution, combined_evolution)
    return casted_evolution


def compose(evolution_0: TimeEvolution, evolution_1: TimeEvolution) -> TimeEvolution:
    """
    2つの時間発展を合成して同一系の時間発展を作る

    Args:
        evolution_0 (TimeEvolution): 合成される側の時間発展
        evolution_1 (TimeEvolution): 合成する側の時間発展

    Returns:
        TimeEvolution: 合成後のユニタリ変換
    """
    composed_matrix = evolution_1.matrix @ evolution_0.matrix
    return TimeEvolution(composed_matrix)


def multiple_compose(evolutions: List[TimeEvolution]) -> TimeEvolution:
    """
    一般的に2つ以上のユニタリ変換を合成して同一系の時間発展を作る

    Args:
        evolutions (List[TimeEvolution]): 合成対象の時間発展のリスト。リストの前方に向かって合成される

    Returns:
        TimeEvolution: 合成後の時間発展
    """
    composed_evolution = evolutions[0]

    for index in range(len(evolutions) - 1):
        composed_evolution = compose(composed_evolution, evolutions[index + 1])

    return composed_evolution
