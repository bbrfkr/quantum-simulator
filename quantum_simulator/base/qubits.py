"""
一般的に混合状態のQubit系に関するクラス群
"""

from typing import List, Tuple

import numpy

from quantum_simulator.base.error import (
    InitializeError,
    InvalidProbabilitiesError,
    NotMatchCountError,
    NotPureError,
    ReductionError,
)
from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits
from quantum_simulator.base.switch_cupy import xp_factory
from quantum_simulator.base.utils import (
    around,
    is_pow2,
    is_probabilities,
    is_real,
    isclose,
)

np = xp_factory()  # typing: numpy


class Qubits:
    """
    一般的に混合状態で複数粒子のQubit系クラス

    Attributes:
        ndarray (np.array): ndarray形式のQubits
        matrix (np.array): 行列形式のQubits
        qubit_count (int): Qubitsに内包されているQubitの数
    """

    def __init__(self, density_array: list):
        """
        Args:
            density_array (list): 密度行列の候補となるリスト。行列形式もしくはndarray形式が許容される
        """
        # arrayの次元をチェック
        tmp_array = np.array(density_array, dtype=complex)
        if not is_qubits_dim(tmp_array):
            message = "[ERROR]: 与えられたリストは形がQubit系に対応しません"
            raise InitializeError(message)

        # 行列表現とndarray表現を導出
        matrix, ndarray = resolve_arrays(tmp_array)
        del tmp_array

        # 固有値と固有ベクトルを導出
        tmp_eigen_values = numpy.linalg.eigvalsh(matrix)

        # 固有値の虚部の有無をチェックし、floatに変換
        if not is_real(tmp_eigen_values):
            message = "[ERROR]: 与えられたリストには虚数の固有値が存在します"
            raise InitializeError(message)

        # 固有値全体が確率分布に対応できるかチェック
        if not is_probabilities(np.real(tmp_eigen_values)):
            message = "[ERROR]: リストから導出された固有値群は確率分布に対応しません"
            raise InitializeError(message)
        del tmp_eigen_values

        # Qubitの個数を導出
        qubit_count = int(len(ndarray.shape) / 2)

        # 初期化
        self.ndarray = ndarray
        self.matrix = matrix
        self.qubit_count = qubit_count
        del ndarray
        del matrix
        del qubit_count

    def __str__(self):
        """
        Qubitsの行列表現に対する文字列を返す

        Returns:
            str: Qubitsの行列表現に対する文字列
        """
        return str(self.matrix)

    def print_ndarray(self):
        """
        Qubitsのndarray表現を出力
        """
        print(self.ndarray)


def is_qubits_dim(array: numpy.array) -> bool:
    """
    与えられたnp.arrayの次元がQubit系を表現する空間の次元たりえるかを判定する

    Args:
        array (np.array): 判定対象のnp.array

    Returns:
        bool: 判定結果
    """

    # 次元のチェック

    # 要素数が2の累乗個であるかチェック
    if not is_pow2(array.size):
        return False

    # array.shapeの要素数をチェック
    len_array_shape = len(array.shape)

    # 2よりも小さい場合、ベクトルであるため、偽
    if len_array_shape < 2:
        return False

    # 2の場合、行列表現とndarray表現の双方の可能性がある
    # いずれの場合も、各要素は一致していなければならない
    # ndarray表現の場合は(2, 2)、つまり2粒子Qubit系でなくてはならない
    # 行列表現の場合は各要素は2の累乗でなければならない
    elif len_array_shape == 2:
        if array.shape[0] != array.shape[1]:
            return False

        for element in array.shape:
            if not is_pow2(element):
                return False

    # 2より大きい場合、ndarray表現にのみ対応する
    # この場合、最低限以下が満たされていなければならない
    # * 要素数が2の倍数であること
    # * 各要素が2であること
    # (各テンソル空間がC^2であることのチェックに対応)
    else:
        if len_array_shape % 2 != 0:
            return False

        for element in array.shape:
            if element != 2:
                return False

    return True


def resolve_arrays(array: numpy.array) -> Tuple[numpy.array, numpy.array]:
    """
    与えられたnp.arrayがQubit系の空間上に存在することを仮定し、その行列形式とndarray形式を導出する

    Args:
        array (np.array): 計算対象のnp.array

    Returns:
        Tuple[numy.array, numy.array]: 行列形式のnumy.arrayとndarray形式のnumy.array
    """
    matrix = None
    ndarray = None

    len_array_shape = len(array.shape)

    # 与えられたarrayが行列表現である場合
    if len_array_shape == 2:
        qubit_count = int(around(np.log2(array.shape[0])))
        ndarray_shape = tuple([2 for i in range(2 * qubit_count)])
        ndarray = array.reshape(ndarray_shape)
        matrix = array

    # 与えられたarrayがndarray表現である場合
    else:
        qubit_count = int(len_array_shape / 2)
        matrix_dim = 2 ** qubit_count
        matrix = array.reshape(matrix_dim, matrix_dim)
        ndarray = array

    return (matrix, ndarray)


def generalize(pure_qubits: PureQubits) -> Qubits:
    """
    与えられたPureQubitsオブジェクトに対応するQubitsオブジェクトを返す

    Args:
        pure_qubits (PureQubits): 一般化対象の純粋状態

    Returns:
        Qubits: 一般化後の純粋状態
    """
    density_array = np.outer(pure_qubits.vector, np.conj(pure_qubits.vector))
    return Qubits(density_array)


def specialize(qubits: Qubits) -> PureQubits:
    """
    純粋状態を表しているQubitsオブジェクトから、対応するPureQubitsオブジェクトを返す

    Args:
        qubits (Qubits): 特殊化対象の純粋状態

    Returns:
        PureQubits: 特殊化後の純粋状態
    """
    # Qubitsが純粋状態かチェックし、対応するインデックスを取り出す
    eigen_values, eigen_vectors = numpy.linalg.eigh(qubits.matrix)

    pure_index = -1
    for index in range(np.size(eigen_values)):
        if isclose(eigen_values[index], 1.0 + 0j):
            pure_index = index

    if pure_index == -1:
        message = "[ERROR]: 対象のQubitsは純粋状態ではありません"
        raise NotPureError(message)

    return PureQubits(eigen_vectors[:, pure_index])


def convex_combination(probabilities: List[float], qubits_list: List[Qubits]) -> Qubits:
    """
    確率のリストとQubitsのリストから凸結合によって、Qubitsオブジェクトを作成する

    Args:
        probabilities (List[float]): 総和が1の正数のリスト
        qubits_list (List[Qubits]): 結合対象のQubitsのリスト

    Returns:
        Qubits: 結合結果としてのQubits
    """

    # 確率リストが確率分布であるかチェック
    if not is_probabilities(np.array(probabilities)):
        message = "[ERROR]: 与えられた確率リストは確率分布ではありません"
        raise InvalidProbabilitiesError(message)

    len_qubits_list = len(qubits_list)

    # 確率リストとQubitsリストの要素数同士が一致するかチェック
    if len_qubits_list != len(probabilities):
        message = "[ERROR]: 与えられた確率リストと純粋状態リストの要素数が一致しません"
        raise NotMatchCountError(message)

    # 密度行列から再度密度行列を導出する
    density_matrix = probabilities[-1] * qubits_list[-1].matrix  # type: numpy.array

    for index in range(len_qubits_list - 1):
        added_matrix = probabilities[index] * qubits_list[index].matrix
        density_matrix = np.add(density_matrix, added_matrix)

    qubits = Qubits(density_matrix)

    del density_matrix
    return qubits


def create_from_ons(probabilities: List[float], ons: OrthogonalSystem) -> Qubits:
    """
    確率のリストと正規直交系からQubitsオブジェクトを作成する

    Args:
        probabilities (List[float]): 総和が1の正数のリスト
        qubits_list (List[Qubits]): 結合対象の正規直交系

    Returns:
        Qubits: 結合結果としてのQubits
    """
    # PureQubitsのgeneralizeリストを作る
    generalized_pure_qubits_list = [
        generalize(pure_qubits) for pure_qubits in ons.qubits_list
    ]

    qubits = convex_combination(probabilities, generalized_pure_qubits_list)

    del generalized_pure_qubits_list
    return qubits


def reduction(target_qubits: Qubits, target_particle: int) -> Qubits:
    """
    指定した系を縮約したQubit系を返す

    Args:
        target_qubits (Qubits): 縮約対象Qubits
        target_particle (int): 縮約対象の系の番号

    Returns:
        Qubits: 縮約後のQubits
    """

    qubit_count = target_qubits.qubit_count

    # 縮約対象が指定された数縮約できるかチェック
    if qubit_count < 2:
        message = "[ERROR]: このQubit系はこれ以上縮約できません"
        raise ReductionError(message)

    # 縮約対象が指定されたQubit番号で縮約できるかチェック
    if target_particle >= qubit_count or target_particle < 0:
        message = "[ERROR]: 指定された要素番号のQubitは存在しません"
        raise ReductionError(message)

    # 縮約の実施
    reduced_array = target_qubits.ndarray

    axis1 = target_particle
    axis2 = target_qubits.qubit_count + target_particle
    reduced_array = np.trace(reduced_array, axis1=axis1, axis2=axis2)

    del target_qubits
    del target_particle
    return Qubits(reduced_array)


def combine(qubits_0: Qubits, qubits_1: Qubits) -> Qubits:
    """
    2つのQubitsを結合した合成系としてのQubitsを作る

    Args:
        qubits_0 (Qubits): 結合される側のQubits
        qubits_1 (Qubits): 結合する側のQubits

    Returns:
        Qubits: 結合結果としてのQubits
    """
    # 新しい状態の生成
    qubits_0_matrix = list(qubits_0.matrix)
    new_matrix = np.vstack(
        tuple(
            [
                np.hstack(
                    tuple([element * qubits_1.matrix for element in qubits_0_row])
                )
                for qubits_0_row in qubits_0_matrix
            ]
        )
    )
    return Qubits(new_matrix)


def multiple_combine(qubits_list: List[Qubits]) -> Qubits:
    """
    一般的に２つ以上のQubits同士を結合する

    Args:
        qubits_list (List[Qubits]): 結合対象のQubitsのリスト

    Returns:
        Qubits: 結合結果としてのQubits
    """
    combined_qubits = qubits_list[0]

    for index in range(len(qubits_list) - 1):
        combined_qubits = combine(combined_qubits, qubits_list[index + 1])

    return combined_qubits


def multiple_reduction(qubits: Qubits, target_particles: List[int]) -> Qubits:
    """
    指定された全ての系を縮約したQubitsを返す

    Args:
        qubits (Qubits): 縮約対象Qubits
        target_particles (List[int]): 縮約対象の系番号のリスト

    Returns:
        Qubits: 縮約後のQubits
    """
    reduced_qubits = qubits
    list.sort(target_particles, reverse=True)

    for target in target_particles:
        reduced_qubits = reduction(reduced_qubits, target)

    return reduced_qubits
