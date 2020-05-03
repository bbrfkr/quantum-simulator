"""
一般的なQubit系の定義
"""

from typing import List, Tuple

import numpy as np
from numpy import linalg as LA

from quantum_simulator.base.conf import APPROX_DIGIT
from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.pure_qubits import PureQubits, is_all_orthogonal


def eig_for_density(
    matrix: np.array, qubit_shape: tuple
) -> Tuple[List[complex], List[PureQubits]]:
    tmp_eigen_values, tmp_eigen_states = LA.eig(matrix)
    eigen_values = [] # type: List[complex]
    eigen_states = [] # type: List[PureQubits]
    for index in range(len(eigen_values)):
        rounded_value = np.round(tmp_eigen_values[index], APPROX_DIGIT)
        if np.equal(rounded_value, 1.0 + 0j):
            eigen_values[index] = 1.0 + 0j
        elif np.equal(rounded_value, 0.0 + 0j):
            eigen_values[index] = 0.0 + 0j
        # 固有ベクトルはPureQubits化
        eigen_states[index] = PureQubits(tmp_eigen_states[:,index].reshape(qubit_shape))

    return (eigen_values, eigen_states)


class Qubits:
    """一般的に複数かつ混合状態のQubit"""

    def __init__(self, probabilities=None, qubits=None, density_array=None):
        array = None
        qubits_count = None
        matrix = None
        matrix_rank = None
        eigen_values = None
        eigen_states = None

        # 確率分布とPureQubits列の候補が与えられた時
        if probabilities is not None and qubits is not None:
            last_qubit = qubits[-1]
            qubits_count = len(qubits)

            # Qubitの数同士が一致しないとエラー
            for index in range(qubits_count - 1):
                if qubits[index].qubit_count != last_qubit.qubit_count:
                    message = "[ERROR]: 与えられたQubit群のQubitの数が一致しません"
                    raise InitializeError(message)

            # Qubitの数と確率の数が一致しないとエラー
            if last_qubit.qubit_count != len(probabilities):
                message = "[ERROR]: 与えられたQubit群の数と確率の数が一致しません"
                raise InitializeError(message)

            probabilities_array = np.array(probabilities)

            # 確率の総和が1でないとエラー
            total_probabilities = np.round(np.sum(probabilities_array), APPROX_DIGIT)
            if total_probabilities != 1.0 + 0j:
                message = "[ERROR]: 与えられた確率の総和が1となりません"
                raise InitializeError(message)

            # 確率として負の値を指定していた場合にもエラー
            if np.any(probabilities_array < 0):
                message = "[ERROR]: 値が負の確率が存在します"
                raise InitializeError(message)

            # ndarray表現と行列表現を算出
            list_arrays = [
                probabilities[index] * qubits[index].projection
                for index in range(qubits_count)
            ]
            array = list_arrays[-1]
            for index in range(len(list_arrays) - 1):
                np.add(array + list_arrays[index])

            matrix_rank = last_qubit.matrix_rank
            matrix = array.reshape(matrix_rank, matrix_rank)

            eigen_values = []  # type: List[complex]
            eigen_states = []  # type: List[PureQubits]
            # まだShatten分解されていない場合はShatten分解を実施
            # 固有値は0または1に近い値は丸める
            if (matrix_rank != qubits_count) or (not is_all_orthogonal(qubits)):
                result_eig = eig_for_density(matrix, last_qubit.amplitudes.shape)
                eigen_values = result_eig[0]
                eigen_states = result_eig[1]
            # 既にShatten分解されている場合は固有値の丸めのみ実施
            else:
                for index in range(len(probabilities)):
                    rounded_value = round(probabilities[index], APPROX_DIGIT)
                    if rounded_value == 1.0:
                        eigen_values[index] == 1.0 + 0j
                    elif rounded_value == 0.0:
                        eigen_values[index] == 0.0 + 0j
                    else:
                        eigen_values[index] == complex(probabilities_array[index])
                eigen_states = qubits

        # 密度行列候補が与えられた時
        elif density_array is not None:
            tmp_array = np.array(density_array)

            # Qubitに対するndarrayもしくは行列になっているかチェック
            size_tmp_array = tmp_array.size
            message = "[ERROR]: 与えられたlistは密度行列に対応しません"

            # ndarrayの時のチェック
            if size_tmp_array != 2:
                # shapeの要素数が2 * qubit_countとなっていないとき
                # -> 行列の次元が2 ** qubit_countとならないとき
                if size_tmp_array % 2 != 0:
                    raise InitializeError(message)

                # shapeの値に2以外の値が含まれる時
                # -> Qubitsを表現するテンソル空間にC^2以外の次元の空間が含まれる時
                else:
                    for shape_element in tmp_array.shape:
                        if shape_element != 2:
                            raise InitializeError(message)
                array = tmp_array
                qubit_count = size_tmp_array / 2
                matrix_rank = 2 ** qubit_count
                matrix_shape = (matrix_rank, matrix_rank)
                matrix = array.reshape(matrix_shape)

            # 行列の時のチェック
            else:
                # 縦横の次元が一致しないか、与えられた行列がベクトルであったときはエラー
                if (
                    (tmp_array.shape[0] != tmp_array.shape[1])
                    and tmp_array.shape[0] < 2
                    and tmp_array.shape[1] < 2
                ):
                    raise InitializeError(message)
                else:
                    tmp_dim = tmp_array.shape[0]

                    # 行列の次元が2の累乗にならない場合はエラー
                    # この時点でQubit数もカウントしておく
                    tmp_qubit_count = 0
                    while tmp_dim % 2 != 1:
                        if tmp_dim % 2 != 0:
                            raise InitializeError(message)
                        tmp_qubit_count += 1
                        tmp_dim /= 2

                    qubit_count = tmp_qubit_count
                    matrix_rank = tmp_array.shape[0]
                matrix = tmp_array

                # ndarrayのshapeを求める
                array_shape = (2 for index in range(2 * qubit_count))
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

        # 初期化
        self.eigen_values = eigen_values
        self.eigen_states = eigen_states
        self.array = array
        self.matrix_rank = matrix_rank
        self.matrix = matrix
        self.qubit_count = qubits_count
