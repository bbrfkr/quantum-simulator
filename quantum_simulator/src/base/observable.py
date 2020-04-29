from math import ceil, sqrt
from random import choices

import numpy as np

from .conf import approx_digit
from .error import InitializeError
from .qubits import Qubits, inner, is_all_orthogonal


class ObservedBasis:
    """観測基底のクラス"""

    def __init__(self, qubits_group: [Qubits]):
        # 観測基底を構成するQubit群同士は互いに直交していなければならない
        if not is_all_orthogonal(qubits_group):
            message = "[ERROR]: 観測基底が直交しません"
            raise InitializeError(message)

        # 観測基底はQubitsの要素数と同じだけ指定されていなければならない
        if len(qubits_group) != qubits_group[0].amplitudes.size:
            message = "[ERROR]: 観測基底を構成するQubitの数が不足しています"
            raise InitializeError(message)

        # 観測基底を構成するqubit列を初期化
        self.qubits_group = qubits_group


class Observable:
    """観測量のクラス"""

    def __init__(self, observed_values: [float], observed_basis: ObservedBasis):
        # 観測値の組と観測基底を構成するQubit群の個数が一致していなければエラー
        len_observed_values = len(observed_values)
        if len_observed_values != len(observed_basis.qubits_group):
            message = "[ERROR]: 観測値の個数と観測基底を構成するQubit群の個数が一致しません"
            raise InitializeError(message)

        # 観測値と観測対象のQubit
        self.elements = [
            {
                "value": observed_values[index],
                "qubits": observed_basis.qubits_group[index],
            }
            for index in range(len_observed_values)
        ]

        # 観測量の表現行列の生成
        elements_arrays = [
            self.elements[index]["value"]
            * np.multiply.outer(
                self.elements[index]["qubits"].amplitudes,
                np.conjugate(self.elements[index]["qubits"].amplitudes),
            )
            for index in range(len_observed_values)
        ]
        array = elements_arrays[-1]
        for index in range(len_observed_values - 1):
            array = np.add(array, elements_arrays[index])

        self.array = array

        # 表現行列を導出する
        matrix_dim = int(sqrt(self.array.size))
        matrix_shape = (matrix_dim, matrix_dim)

        self.matrix_shape = matrix_shape
        self.matrix = self.array.reshape(matrix_shape)

    def __str__(self):
        """観測量の二次元行列表現を出力"""
        return str(self.matrix)

    def print_array(self):
        """観測量のndarray表現を出力"""
        print(str(self.array))

    def expected_value(self, target: Qubits) -> float:
        """対象Qubit群に対する観測量の期待値を返す"""
        expected_value = 0
        for element in self.elements:
            expected_value += (
                element["value"] * abs(inner(element["qubits"], target)) ** 2
            )

        return expected_value

    def observe(self, target: Qubits) -> float:
        """観測を実施して観測値を取得し、Qubits群を収束させる"""
        observed_probabilities = [
            abs(inner(self.elements[index]["qubits"], target)) ** 2
            for index in range(len(self.elements))
        ]
        observed_result = choices(self.elements, observed_probabilities)[0]

        # 観測値によって識別されたQubitsを選択する
        observed_qubits_group = []
        for element in self.elements:
            if observed_result["value"] == element["value"]:
                observed_qubits_group.append(element["qubits"])

        # 選択されたQubits群で射影作用素を作る
        len_observed_qubits_group = len(observed_qubits_group)
        projection = observed_qubits_group[-1].projection()
        for index in range(len_observed_qubits_group - 1):
            projection = np.add(projection, observed_qubits_group[index].projection())

        # 射影作用素とQubit群をそれぞれ二次元行列、ベクトルに変換
        matrix_dim = target.amplitudes.size
        proj_matrix = projection.reshape(matrix_dim, matrix_dim)
        target_vector = target.amplitudes.reshape(target.amplitudes.size)

        # 観測によるQubitの収束 - 射影の適用と規格化
        post_vector = np.dot(proj_matrix, target_vector)
        norm_post_vector = np.linalg.norm(post_vector)
        normalized_post_vector = ((1.0 / norm_post_vector) * post_vector).reshape(target.amplitudes.shape)
        target.amplitudes = normalized_post_vector

        # 観測値の返却
        return observed_result["value"]
