"""
一般的なQubit系の定義
"""

from typing import List, Tuple

import numpy as np
from numpy import linalg as LA

from quantum_simulator.base.conf import APPROX_DIGIT
from quantum_simulator.base.error import InitializeError, ReductionError
from quantum_simulator.base.pure_qubits import PureQubits, is_all_orthogonal
from quantum_simulator.base import pure_qubits


def eig_for_density(
    matrix: np.array, qubit_shape: tuple
) -> Tuple[List[complex], List[PureQubits]]:
    """密度行列から固有値、固有状態を導出する"""

    tmp_eigen_values, tmp_eigen_states = LA.eig(matrix)
    eigen_values = []  # type: List[complex]
    eigen_states = []  # type: List[PureQubits]
    for index in range(len(tmp_eigen_values)):

        # 固有値は0または1に近い値は丸める
        rounded_value = np.round(tmp_eigen_values[index], APPROX_DIGIT)
        if np.equal(rounded_value, 1.0 + 0j):
            eigen_values.append(1.0 + 0j)
        elif np.equal(rounded_value, 0.0 + 0j):
            eigen_values.append(0.0 + 0j)
        else:
            eigen_values.append(complex(tmp_eigen_values[index]))

        # 固有ベクトルはPureQubits化
        eigen_states.append(PureQubits(tmp_eigen_states[:, index].reshape(qubit_shape)))

    return (eigen_values, eigen_states)


class Qubits:
    """一般的に複数かつ混合状態のQubit"""

    def __init__(self, probabilities=None, qubits=None, density_array=None):
        array = None
        qubit_count = None
        matrix = None
        matrix_dim = None
        eigen_values = None
        eigen_states = None
        probabilities_array = None

        # 確率分布とPureQubits列の候補が与えられた時
        if probabilities is not None and qubits is not None:
            last_qubit = qubits[-1]
            qubits_count = len(qubits)

            # Qubit群のQubit数同士が一致しないとエラー
            for index in range(qubits_count - 1):
                if qubits[index].qubit_count != last_qubit.qubit_count:
                    message = "[ERROR]: 与えられたQubit群のQubitの数が一致しません"
                    raise InitializeError(message)

            qubit_count = last_qubit.qubit_count

            # Qubit群の数と確率の数が一致しないとエラー
            if qubits_count != len(probabilities):
                message = "[ERROR]: 与えられたQubit群の数と確率の数が一致しません"
                raise InitializeError(message)

            # ndarray表現と行列表現を算出
            list_arrays = [
                probabilities[index] * qubits[index].projection
                for index in range(qubits_count)
            ]
            array = list_arrays[-1]
            for index in range(len(list_arrays) - 1):
                array = np.add(array, list_arrays[index])

            matrix_dim = last_qubit.projection_matrix_dim
            matrix = array.reshape(matrix_dim, matrix_dim)

            eigen_values = []  # type: List[complex]
            eigen_states = []  # type: List[PureQubits]

            # まだShatten分解されていない場合はShatten分解を実施
            if (matrix_dim != qubits_count) or (not is_all_orthogonal(qubits)):
                result_eig = eig_for_density(matrix, last_qubit.array.shape)
                eigen_values = result_eig[0]
                eigen_states = result_eig[1]

            # 既にShatten分解されている場合は固有値の丸めのみ実施
            else:
                for index in range(len(probabilities)):
                    rounded_value = np.round(probabilities[index], APPROX_DIGIT)
                    if rounded_value == 1.0:
                        eigen_values.append(1.0 + 0j)
                    elif rounded_value == 0.0:
                        eigen_values.append(0.0 + 0j)
                    else:
                        eigen_values.append(complex(probabilities[index]))
                eigen_states = qubits

        # 密度行列候補が与えられた時
        elif density_array is not None:
            tmp_array = np.array(density_array, dtype=complex)

            # Qubitに対するndarrayもしくは行列になっているかチェック
            len_tmp_array_shape = len(tmp_array.shape)
            message = "[ERROR]: 与えられたlistはQubit系の密度行列に対応しません"

            # ndarrayの時のチェック
            if len_tmp_array_shape != 2:
                # shapeの要素数が2 * qubit_countとなっていないとき
                # -> 行列の次元が2 ** qubit_countとならないとき
                if len_tmp_array_shape % 2 != 0:
                    raise InitializeError(message)

                # shapeの値に2以外の値が含まれる時
                # -> Qubitsを表現するテンソル空間にC^2以外の次元の空間が含まれる時
                else:
                    for shape_element in tmp_array.shape:
                        if shape_element != 2:
                            raise InitializeError(message)
                array = tmp_array

                # shapeの要素数が2の倍数でないときは、行列に対応させられないためエラー
                if len_tmp_array_shape % 2 != 0:
                    raise InitializeError(message)

                qubit_count = int(len_tmp_array_shape / 2)
                matrix_dim = 2 ** qubit_count
                matrix_shape = (matrix_dim, matrix_dim)
                matrix = array.reshape(matrix_shape)

            # 行列の時のチェック
            else:
                # 縦横の次元が一致しないか、与えられた行列がベクトルであったときはエラー
                if (
                    (tmp_array.shape[0] != tmp_array.shape[1])
                    or tmp_array.shape[0] < 2
                    or tmp_array.shape[1] < 2
                ):
                    raise InitializeError(message)

                else:
                    tmp_dim = tmp_array.shape[0]
                    tmp_qubit_count = 0

                    # 行列の次元が2の累乗にならない場合はエラー
                    # この時点でQubit数もカウントしておく
                    while True:
                        if tmp_dim % 2 != 0:
                            raise InitializeError(message)
                        tmp_qubit_count += 1
                        tmp_dim /= 2
                        if tmp_dim % 2 == 1:
                            break

                    qubit_count = int(tmp_qubit_count)
                    matrix_dim = tmp_array.shape[0]

                matrix = tmp_array

                # ndarrayのshapeを求める
                array_shape = tuple([2 for index in range(2 * qubit_count)])
                array = matrix.reshape(array_shape)

            # 固有値、固有ベクトルを計算
            pure_qubit_shape = tuple([2 for index in range(qubit_count)])
            result_eig = eig_for_density(matrix, pure_qubit_shape)
            eigen_values = result_eig[0]
            eigen_states = result_eig[1]

        # 必須パラメータの指定なしの場合、エラー
        else:
            message = "[ERROR]: 確率のリストとPureQubitsのリスト、もしくは密度行列のndarrayが必須です"
            raise InitializeError(message)

        probabilities_array = np.array(eigen_values, dtype=complex)

        # 確率(固有値)の総和が1でないとエラー
        total_probabilities = np.round(np.sum(probabilities_array), APPROX_DIGIT)
        if total_probabilities != 1.0 + 0j:
            message = "[ERROR]: 与えられた確率の総和が1となりません"
            raise InitializeError(message)

        # 確率として負の値を指定していた場合にもエラー
        if np.any(np.real(probabilities_array) < 0):
            message = "[ERROR]: 値が負の確率が存在します"
            raise InitializeError(message)

        # 初期化
        self.eigen_values = eigen_values
        self.eigen_states = eigen_states
        self.array = array
        self.matrix_dim = matrix_dim
        self.matrix = matrix
        self.qubit_count = qubit_count

    def __str__(self):
        """Qubitsの行列表現を出力"""
        return str(self.matrix)

    def print_array(self):
        """Qubitsのndarray表現を出力"""
        print(self.array)


def reduction(target_qubits: Qubits, target_particle: int) -> Qubits:
    """target番目のQubitを縮約した局所Qubit群を返す"""
    if target_qubits.qubit_count == 1:
        message = "[ERROR]: このQubit系はこれ以上縮約できません"
        raise ReductionError(message)

    # TODO target_listの長さや中の値はもっとバリデーションすべき

    axis1 = target_particle
    axis2 = target_qubits.qubit_count + target_particle
    reduced_array = np.trace(target_qubits.array, axis1=axis1, axis2=axis2,)

    return Qubits(density_array=reduced_array)


def combine(qubits_0: Qubits, qubits_1: Qubits) -> Qubits:
    probabilities = [
        eigen_value_0 * eigen_value_1
        for eigen_value_0 in qubits_0.eigen_values
        for eigen_value_1 in qubits_1.eigen_values
    ]
    eigen_states = [
        pure_qubits.combine(eigen_state_0, eigen_state_1)
        for eigen_state_0 in qubits_0.eigen_states
        for eigen_state_1 in qubits_1.eigen_states
    ]
    new_qubits = Qubits(probabilities, eigen_states)
    return new_qubits
