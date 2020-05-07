"""
一般的なQubit系の定義
"""

from typing import List, Tuple, Union

import numpy as np
from numpy import linalg as LA

from quantum_simulator.base import pure_qubits
from quantum_simulator.base.error import (
    InitializeError,
    InvalidProbabilitiesError,
    NotMatchCountError,
    ReductionError,
    NotPureError
)
from quantum_simulator.base.pure_qubits import PureQubits, OrthogonalBasis
from quantum_simulator.base.utils import (
    around,
    is_pow2,
    is_probabilities,
    is_real,
    isclose,
)


class Qubits:
    """
    一般的に複数かつ混合状態のQubit群のクラス
        eigen_values: 固有値のリスト
        eigen_states: 固有状態のリスト
        ndarray: ndarray形式のQubits
        matrix: 行列形式のQubits
        matrix_dim: 行列形式における行列の次元
        qubit_count: 内包されているQubitの数
    """

    def __init__(self, density_array: list):
        """
        初期化
            density_array: 密度行列に対応するnp.arrayインスタンス。
                           行列形式とndarray形式を許容する。
        """

        # arrayの次元をチェック
        tmp_array = np.array(density_array, dtype=complex)
        if not is_qubits_dim(tmp_array):
            message = "[ERROR]: 与えられたリストは形がQubit系に対応しません"
            raise InitializeError(message)

        # 行列表現とndarray表現を導出
        matrix, ndarray = resolve_arrays(tmp_array)

        # 固有値と固有ベクトルを導出
        eigen_values, eigen_states = resolve_eigen(matrix)

        # 固有値の虚部の有無をチェックし、floatに変換
        if not is_real(eigen_values):
            message = "[ERROR]: 与えられたリストには虚数の固有値が存在します"
            raise InitializeError(message)

        eigen_values = np.real(eigen_values)

        # 固有値全体が確率分布に対応できるかチェック
        if not is_probabilities(eigen_values):
            message = "[ERROR]: リストから導出された固有値群は確率分布に対応しません"
            raise InitializeError(message)

        # 行列の次元を導出
        matrix_dim = matrix.shape[0]

        # Qubitの個数を導出
        qubit_count = int(len(ndarray.shape) / 2)

        # 初期化
        self.eigen_values = eigen_values
        self.eigen_states = eigen_states
        self.ndarray = ndarray
        self.matrix = matrix
        self.matrix_dim = matrix_dim
        self.qubit_count = qubit_count

    def __str__(self):
        """Qubitsの行列表現を出力"""
        return str(self.matrix)

    def print_ndarray(self):
        """Qubitsのndarray表現を出力"""
        print(self.ndarray)

    def is_pure(self) -> bool:
        """Qubitsが純粋状態か判定する"""
        for eigen_value in self.eigen_values:
            if isclose(eigen_value, 1.0 + 0j):
                return True

        return False


def is_qubits_dim(array: np.array) -> bool:
    """与えられたarrayの形がQubit系を表現しているか判定する"""

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


def resolve_arrays(array: np.array) -> Tuple[np.array, np.array]:
    """
    Qubit系の空間上のnp.arrayを仮定し、行列表現とndarray表現を導出して返す
    """
    matrix = None
    ndarray = None

    len_array_shape = len(array.shape)

    # 与えられたarrayが行列表現である場合
    if len_array_shape == 2:
        qubit_count = int(np.log2(array.shape[0]))
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


def resolve_eigen(matrix: np.array) -> Tuple[List[complex], List[PureQubits]]:
    """
    行列表現のarrayを仮定し、固有値・固有状態を導出する
    """

    # 固有値・固有状態の導出
    tmp_eigen_values, tmp_eigen_states = LA.eig(matrix)

    # 実際に呼び出し元に渡すオブジェクトの整理
    eigen_values = []  # type: List[complex]
    eigen_states = []  # type: List[PureQubits]
    for index in range(len(tmp_eigen_values)):

        # 固有値は0または1に近い値は丸める
        rounded_value = around(tmp_eigen_values[index])
        if rounded_value == 1.0 + 0j:
            eigen_values.append(1.0 + 0j)
        elif rounded_value == 0.0 + 0j:
            eigen_values.append(0.0 + 0j)
        else:
            eigen_values.append(complex(tmp_eigen_values[index]))

        # 固有ベクトルはPureQubits化
        eigen_states.append(PureQubits(tmp_eigen_states[:, index]))

    return (eigen_values, eigen_states)


def generalize(pure_qubits: PureQubits) -> Qubits:
    """PureQubitsから対応するQubitsを作る"""
    density_array = pure_qubits.projection
    return Qubits(density_array)


def specialize(qubits: Qubits) -> PureQubits:
    """固有値1を持つQubitsから対応するPureQubitsを作る"""
    # Qubitsが純粋状態かチェックし、対応するインデックスを取り出す
    pure_index = -1
    for index in range(len(qubits.eigen_values)):
        if isclose(qubits.eigen_values[index], 1.0 + 0j):
            pure_index = index

    if pure_index == -1:
        message = "[ERROR]: 対象のQubitsは純粋状態ではありません"
        raise NotPureError(message)

    return qubits.eigen_states[pure_index]


def convex_combination(
    probabilities: List[float], qubits_list: List[Union[PureQubits, Qubits]]
) -> Qubits:
    """確率リストと(Pure)QubitsリストからQubitsオブジェクトを作成する"""

    # 確率リストが確率分布であるかチェック
    if not is_probabilities(probabilities):
        message = "[ERROR]: 与えられた確率リストは確率分布ではありません"
        raise InvalidProbabilitiesError(message)

    len_qubits_list = len(qubits_list)

    # 確率リストと純粋状態リストの要素数同士が一致するかチェック
    if len_qubits_list != len(probabilities):
        message = "[ERROR]: 与えられた確率リストと純粋状態リストの要素数が一致しません"
        raise NotMatchCountError(message)

    # 密度行列から再度密度行列を導出する
    density_matrix = None

    if isinstance(qubits_list[-1], PureQubits):
        density_matrix = probabilities[-1] * qubits_list[-1].projection_matrix
    else:
        density_matrix = probabilities[-1] * qubits_list[-1].matrix

    for index in range(len_qubits_list - 1):
        added_matrix = None

        if isinstance(qubits_list[index], PureQubits):
            added_matrix = probabilities[index] * qubits_list[index].projection_matrix
        else:
            added_matrix = probabilities[index] * qubits_list[index].matrix

        density_matrix = np.add(density_matrix, added_matrix)

    qubits = Qubits(density_matrix)
    return qubits


def create_from_basis(
    probabilities: List[float], basis: OrthogonalBasis
) -> Qubits:
    """確率リストと直交基底からQubitsオブジェクトを作成する"""
    qubits = convex_combination(probabilities, basis.qubits_list)
    return qubits


def reduction(target_qubits: Union[Qubits, PureQubits], target_particle: int) -> Qubits:
    """target番目のQubitを縮約した局所Qubit群を返す"""

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
    if isinstance(target_qubits, PureQubits):
        reduced_array = target_qubits.projection
    else:
        reduced_array = target_qubits.ndarray

    axis1 = target_particle
    axis2 = target_qubits.qubit_count + target_particle
    reduced_array = np.trace(reduced_array, axis1=axis1, axis2=axis2)

    return Qubits(reduced_array)


def combine(
    qubit_0: Union[Qubits, PureQubits], qubit_1: Union[Qubits, PureQubits]
) -> Qubits:
    """２つのQubit系を結合して新たなQubit系を作る"""
    # 純粋状態を考慮し、結合する情報を整理
    eigen_values_0 = None
    eigen_states_0 = None
    eigen_values_1 = None
    eigen_states_1 = None

    if isinstance(qubit_0, PureQubits):
        eigen_values_0 = [1.0]
        eigen_states_0 = [qubit_0]
    else:
        eigen_values_0 = qubit_0.eigen_values
        eigen_states_0 = qubit_0.eigen_states
    if isinstance(qubit_1, PureQubits):
        eigen_values_1 = [1.0]
        eigen_states_1 = [qubit_1]
    else:
        eigen_values_1 = qubit_1.eigen_values
        eigen_states_1 = qubit_1.eigen_states

    # 確率分布の結合
    probabilities = [
        value_0 * value_1 for value_1 in eigen_values_1 for value_0 in eigen_values_0
    ]

    # 固有状態の結合
    eigen_states = [
        pure_qubits.combine(state_0, state_1)
        for state_1 in eigen_states_1
        for state_0 in eigen_states_0
    ]

    # 新しい状態の生成
    new_qubits = convex_combination(probabilities, eigen_states)
    return new_qubits
