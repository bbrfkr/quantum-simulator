"""
一般的なQubit系の定義
"""

from typing import overload, List
from quantum_simulator.base.pure_qubits import PureQubits, is_all_orthogonal
from quantum_simulator.base.error import InitializeError
import numpy as np
from numpy import linalg as la
from quantum_simulator.base.conf import APPROX_DIGIT

def eig_for_density(matrix: np.array, qubit_shape: tuple) -> ([complex], [PureQubits]):
    eigen_values, eigen_states = la.eig(matrix)
    for index in range(len(eigen_values)):
        rounded_value = np.round(eigen_values[index], APPROX_DIGIT)
        if np.equal(rounded_value, 1.0 + 0j):
            eigen_values[index] = 1.0 + 0j
        elif np.equal(rounded_value, 0.0 + 0j):
            eigen_values[index] = 0.0 + 0j
        # 固有ベクトルはPureQubits化
        eigen_states[index] = PureQubits(eigen_states[index].reshape(qubit_shape))

    return (eigen_values, eigen_states)

class Qubits:
    """一般的に複数かつ混合状態のQubit"""
    @overload
    def __init__(self, density_array: list):
        array = None
        qubits_count = None
        matrix = None
        matrix_dim = None
        eigen_values = None
        eigen_states = None

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
            matrix_dim = 2 ** qubit_count
            matrix_shape = (matrix_dim, matrix_dim)
            matrix = array.reshape(matrix_shape)

        # 行列の時のチェック
        else:
            # 縦横の次元が一致しないか、与えられた行列がベクトルであったときはエラー
            if (tmp_array.shape[0] != tmp_array.shape[1]) and tmp_array.shape[0] < 2 and tmp_array.shape[1] < 2:
                raise InitializeError(message)
            else:
                tmp_dim = tmp_array.shape[0]

                # 行列の次元が2の累乗にならない場合はエラー
                # この時点でQubit数もカウントしておく
                tmp_qubit_count = 0
                while(tmp_dim % 2 != 1):
                    if tmp_dim % 2 != 0:
                        raise InitializeError(message)
                    tmp_qubit_count += 1
                    tmp_dim /= 2

                qubit_count = tmp_qubit_count
                matrix_dim = tmp_array.shape[0]
            matrix = tmp_array
            
            # ndarrayのshapeを求める
            array_shape = (2 for index in range(2 * qubit_count))
            array = matrix.reshape(array_shape)

        # 固有値、固有ベクトルを計算
        pure_qubit_shape = (2 for index in range(qubit_count))
        eigen_values, eigen_states = eig_for_density(matrix, pure_qubit_shape)

        # 初期化
        self.eigen_values = eigen_values
        self.eigen_states = eigen_states
        self.array = array
        self.matrix_dim = matrix_dim
        self.matrix = matrix
        self.qubit_count = qubits_count


    def __init__(self, probabilities: List[float], qubits: List[PureQubits]):
        last_qubit = qubits[-1]
        qubits_count = len(qubits)

        # Qubitの数同士が一致しないとエラー
        for index in range(qubits_count - 1):
            if  qubits[index].qubit_count != last_qubit.qubit_count:
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
        array = np.add([probabilities[index] * qubits[index].amplitudes for index in range(qubits_count)])
        matrix_dim = last_qubit.matrix_dim
        matrix = array.reshape(matrix_dim, matrix_dim)

        eigen_values = []
        eigen_states = []
        # まだShatten分解されていない場合はShatten分解を実施
        # 固有値は0または1に近い値は丸める
        if (matrix_dim != qubits_count) or (not is_all_orthogonal(qubits)):
            eigen_values, eigen_states = eig_for_density(matrix, last_qubit.shape)

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
        
        # 初期化
        self.eigen_values = eigen_values
        self.eigen_states = eigen_states
        self.array = array
        self.matrix_dim = matrix_dim
        self.matrix = matrix
        self.qubit_count = qubits_count
