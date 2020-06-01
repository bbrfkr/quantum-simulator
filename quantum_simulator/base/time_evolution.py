"""
時間発展を記述するクラス群
"""

from typing import List

from quantum_simulator.base.error import (
    IncompatibleDimensionError,
    InitializeError,
    NotCompleteError,
)
from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits, combine_ons
from quantum_simulator.base.qubits import Qubits, is_qubits_dim, resolve_arrays
from quantum_simulator.base.switch_cupy import xp_factory
from quantum_simulator.base.utils import allclose

np = xp_factory()  # typing: numpy


class TimeEvolution:
    """
    ユニタリ変換による時間発展のクラス

    Attributes:
        ndarray: ndarray形式のユニタリ変換
        matrix: 行列形式のユニタリ変換
        matrix_dim: ユニタリ変換の行列の次元
    """

    def __init__(self, unitary_array: list):
        """
        Args:
            unitary_array: ユニタリ変換の候補となるリスト。行列形式とndarray形式を許容する
        """
        tmp_array = np.array(unitary_array, dtype=complex)

        # 次元のチェック
        if not is_qubits_dim(tmp_array):
            message = "[ERROR]: 与えられたリストはQubit系上の作用素ではありません"
            raise InitializeError(message)

        # 行列表現とndarray表現の導出
        matrix, ndarray = resolve_arrays(tmp_array)
        matrix_dim = matrix.shape[0]
        # ユニタリ性のチェック
        hermite_matrix = np.dot(matrix, np.conj(matrix.T))
        if not allclose(hermite_matrix, np.identity(matrix_dim)):
            message = "[ERROR]: 与えられたリストはユニタリ変換ではありません"
            raise InitializeError(message)

        # 初期化
        self.ndarray = ndarray
        self.matrix = matrix
        self.matrix_dim = matrix_dim

    def __str__(self):
        """
        ユニタリ変換の行列表現の文字列を返す

        Returns:
            str: ユニタリ変換の行列表現の文字列
        """
        return str(self.matrix)

    def print_ndarray(self):
        """
        ユニタリ変換のndarray表現を出力する
        """
        print(str(self.array))

    def operate(self, qubits: Qubits) -> Qubits:
        """
        対象Qubitsを時間発展によって別のQubitsに変換し、変換後のQubitsを返す

        Args:
            qubits (Qubits): 変換対象のQubits

        Returns:
            Qubits: 変換後のQubits
        """
        if self.matrix_dim != qubits.matrix_dim:
            message = "[ERROR]: 変換対象のQubit数が不正です"
            raise IncompatibleDimensionError(message)

        transformed_matrix = np.dot(
            np.dot(self.matrix, qubits.matrix), np.conj(self.matrix.T)
        )

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
        message = "[ERROR]: 正規直交基底を指定してください"
        raise NotCompleteError(message)

    # 観測基底を構成するQubit群の個数同士が一致していなければエラー
    len_pre_ons = len(pre_ons.qubits_list)
    if len_pre_ons != len(post_ons.qubits_list):
        message = "[ERROR]: 変換後のQubit数が変換前と異なります"
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

    return TimeEvolution(matrix)


def combine(evolution_0: TimeEvolution, evolution_1: TimeEvolution) -> TimeEvolution:
    """
    2つの時間発展を結合して合成系の時間発展を作る

    Args:
        evolution_0 (TimeEvolution): 結合される側の時間発展
        evolution_1 (TimeEvolution): 結合する側の時間発展

    Returns:
        TimeEvolution: 結合後の時間発展
    """
    # 各ユニタリ行列を標準基底からの基底変換とみなして、ONBを抽出する
    matrix_0 = evolution_0.matrix
    matrix_0_dim = evolution_0.matrix_dim
    onb_0 = OrthogonalSystem(
        [
            PureQubits(list(np.conj(np.conj(matrix_0)[:, index])))
            for index in range(matrix_0_dim)
        ]
    )
    matrix_1 = evolution_1.matrix
    matrix_1_dim = evolution_1.matrix_dim
    onb_1 = OrthogonalSystem(
        [
            PureQubits(list(np.conj(np.conj(matrix_1)[:, index])))
            for index in range(matrix_1_dim)
        ]
    )

    # 合成系における変換後の基底を求める
    post_onb = combine_ons(onb_0, onb_1)

    # 合成系における変換前の基底である標準基底を求める
    identity = np.identity(matrix_0_dim * matrix_1_dim)
    pre_onb = OrthogonalSystem(
        [
            PureQubits(list(identity[:, index]))
            for index in range(matrix_0_dim * matrix_1_dim)
        ]
    )

    new_evolution = create_from_onb(pre_onb, post_onb)
    return new_evolution


def multiple_combine(evolutions: List[TimeEvolution]) -> TimeEvolution:
    """
    一般的に2つ以上のユニタリ変換を結合して合成系の時間発展を作る

    Args:
        evolutions (List[TimeEvolution]): 結合対象の時間発展のリスト

    Returns:
        TimeEvolution: 結合後の時間発展
    """
    combined_evolution = evolutions[0]

    for index in range(len(evolutions) - 1):
        combined_evolution = combine(combined_evolution, evolutions[index + 1])

    return combined_evolution


def compose(evolution_0: TimeEvolution, evolution_1: TimeEvolution) -> TimeEvolution:
    """
    2つの時間発展を合成して同一系の時間発展を作る

    Args:
        evolution_0 (TimeEvolution): 合成される側の時間発展
        evolution_1 (TimeEvolution): 合成する側の時間発展

    Returns:
        TimeEvolution: 合成後のユニタリ変換
    """
    composed_matrix = np.dot(evolution_1.matrix, evolution_0.matrix)
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
