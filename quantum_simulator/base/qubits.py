"""
一般的なQubit系の定義
"""

from typing import List, Tuple

import numpy as np
from numpy import linalg as LA

from quantum_simulator.base import pure_qubits
from quantum_simulator.base.error import InitializeError, ReductionError
from quantum_simulator.base.pure_qubits import PureQubits, all_orthogonal
from quantum_simulator.base.utils import is_pow2, around


class Qubits:
    """
    一般的に複数かつ混合状態のQubit
        eigen_values: 固有値
        eigen_states: 固有状態
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

        # 負の固有値の存在をチェック
        if np.any(around(np.array(eigen_values)) < 0):
            message = "[ERROR]: 与えられたリストには負の固有値が存在します"
            raise InitializeError(message)

        # 固有値の総和(トレース)が1であるかチェック
        if around(np.sum(np.array(eigen_values))) != 1:
            message = "[ERROR]: 与えられたリストはトレースが1ではありません"
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


def is_qubits_dim(array: np.array) -> bool:
    """与えられたarrayの形がQubit系を表現しているか判定する"""

    # 次元のチェック
    ## 要素数が2の累乗個であるかチェック
    if not is_pow2(array.size):
        return False

    ## array.shapeの要素数をチェック
    len_array_shape = len(array.shape)

    ### 2よりも小さい場合、ベクトルであるため、偽
    if len_array_shape < 2:
        return False

    ### 2の場合、行列表現とndarray表現の双方の可能性がある
    ### いずれの場合も、各要素は一致していなければならない
    ### ndarray表現の場合は(2, 2)、つまり2粒子Qubit系でなくてはならない
    ### 行列表現の場合は各要素は2の累乗でなければならない
    elif len_array_shape == 2:
        if array.shape[0] != array.shape[1]:
            return False

        for element in array.shape:
            if not is_pow2(element):
                return False

    ### 2より大きい場合、ndarray表現にのみ対応する
    ### この場合、最低限以下が満たされていなければならない
    ### * 要素数が2の倍数であること
    ### * 各要素が2であること
    ### (各テンソル空間がC^2であることのチェックに対応)
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


# def create_from_distribution(probabilities: List[float], qubits: List[PureQubits]) -> Qubits:
#     """確率リストと純粋状態リストからQubitsインスタンスを作成する"""
#     return None


# def reduction(target_qubits: Qubits, target_particle: int) -> Qubits:
#     """target番目のQubitを縮約した局所Qubit群を返す"""
#     if target_qubits.qubit_count == 1:
#         message = "[ERROR]: このQubit系はこれ以上縮約できません"
#         raise ReductionError(message)

#     # TODO target_listの長さや中の値はもっとバリデーションすべき

#     axis1 = target_particle
#     axis2 = target_qubits.qubit_count + target_particle
#     reduced_array = np.trace(target_qubits.array, axis1=axis1, axis2=axis2)

#     return Qubits(density_array=reduced_array)


# def combine(qubits_0: Qubits, qubits_1: Qubits) -> Qubits:
#     probabilities = [
#         eigen_value_0 * eigen_value_1
#         for eigen_value_0 in qubits_0.eigen_values
#         for eigen_value_1 in qubits_1.eigen_values
#     ]
#     eigen_states = [
#         pure_qubits.combine(eigen_state_0, eigen_state_1)
#         for eigen_state_0 in qubits_0.eigen_states
#         for eigen_state_1 in qubits_1.eigen_states
#     ]
#     new_qubits = create_from_distribution(probabilities, eigen_states)
#     return new_qubits
