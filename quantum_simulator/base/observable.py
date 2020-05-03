"""
観測量に関するクラス群
"""

from math import sqrt
from random import choices
from typing import List

import numpy as np

from .error import InitializeError
from .pure_qubits import PureQubits, inner, is_all_orthogonal
from .typing import ObservableElements


class ObservedBasis:  # pylint: disable=too-few-public-methods
    """観測基底のクラス"""

    def __init__(self, qubits_group: List[PureQubits]):
        # 観測基底を構成するQubit群同士は互いに直交していなければならない
        if not is_all_orthogonal(qubits_group):
            message = "[ERROR]: 観測基底が直交しません"
            raise InitializeError(message)

        # 観測基底はPureQubitsの要素数と同じだけ指定されていなければならない
        if len(qubits_group) != qubits_group[0].amplitudes.size:
            message = "[ERROR]: 観測基底を構成するQubitの数が不足しています"
            raise InitializeError(message)

        # 観測基底を構成するqubit列を初期化
        self.qubits_group = qubits_group


class Observable:  # pylint: disable=too-few-public-methods
    """観測量のクラス"""

    elements: ObservableElements
    observed_values: List[float]
    observed_basis: ObservedBasis

    def __init__(self, observed_values: List[float], observed_basis: ObservedBasis):
        # 観測値の組と観測基底を構成するQubit群の個数が一致していなければエラー
        len_observed_values = len(observed_values)
        if len_observed_values != len(observed_basis.qubits_group):
            message = "[ERROR]: 観測値の個数と観測基底を構成するQubit群の個数が一致しません"
            raise InitializeError(message)

        self.observed_values = observed_values
        self.observed_basis = observed_basis

        # 観測値と観測対象のQubit
        self.elements = [
            {
                "value": observed_values[index],
                "qubits": observed_basis.qubits_group[index],
            }
            for index in range(len_observed_values)
        ]

        # 観測量のndarrayの生成
        elements_arrays = [
            self.elements[index]["value"]
            * np.multiply.outer(
                self.elements[index]["qubits"].amplitudes,
                np.conjugate(self.elements[index]["qubits"].amplitudes),
            )
            for index in range(len_observed_values)
        ]
        array = elements_arrays[-1]
        for index in range(len_observed_values - 1):  # pylint: disable=R0801
            array = np.add(array, elements_arrays[index])

        self.array = array

        # 表現行列を導出する
        matrix_dim = int(sqrt(self.array.size))

        self.matrix_dim = matrix_dim
        self.matrix_shape = (self.matrix_dim, self.matrix_dim)
        self.matrix = self.array.reshape(self.matrix_shape)

    def __str__(self):
        """観測量の二次元行列表現を出力"""
        return str(self.matrix)

    def print_array(self):
        """観測量のndarray表現を出力"""
        print(str(self.array))

    def expected_value(self, target: PureQubits) -> float:
        """対象Qubit群に対する観測量の期待値を返す"""
        expected_value = 0.0
        for element in self.elements:
            expected_value += (
                element["value"] * abs(inner(element["qubits"], target)) ** 2
            )

        return expected_value

    def observe(self, target: PureQubits) -> float:
        """観測を実施して観測値を取得し、PureQubits群を収束させる"""
        observed_probabilities = [
            abs(inner(self.elements[index]["qubits"], target)) ** 2
            for index in range(len(self.elements))
        ]
        observed_result = choices(self.elements, observed_probabilities)[0]

        # 観測値によって識別されたPureQubitsを選択する
        observed_pure_qubits_group = []
        for element in self.elements:
            if observed_result["value"] == element["value"]:
                observed_pure_qubits_group.append(element["qubits"])

        # 選択されたPureQubits群で射影作用素を作る
        len_observed_pure_qubits_group = len(observed_pure_qubits_group)
        projection = observed_pure_qubits_group[-1].projection
        for index in range(len_observed_pure_qubits_group - 1):
            projection = np.add(
                projection, observed_pure_qubits_group[index].projection
            )

        # 射影作用素とQubit群をそれぞれ二次元行列、ベクトルに変換
        matrix_dim = target.amplitudes.size
        proj_matrix = projection.reshape(matrix_dim, matrix_dim)
        target_vector = target.amplitudes.reshape(target.amplitudes.size)

        # 観測によるQubitの収束 - 射影の適用と規格化
        post_vector = np.dot(proj_matrix, target_vector)
        norm_post_vector = np.linalg.norm(post_vector)
        normalized_post_vector = ((1.0 / norm_post_vector) * post_vector).reshape(
            target.amplitudes.shape
        )
        target.amplitudes = normalized_post_vector

        # 観測値の返却
        return observed_result["value"]


def combine_basis(basis_0: ObservedBasis, basis_1: ObservedBasis) -> ObservedBasis:
    """二つの観測基底から合成系の観測基底を作る"""
    new_observed_basis = ObservedBasis(
        [
            PureQubits(np.tensordot(qubits_0.amplitudes, qubits_1.amplitudes, 0))
            for qubits_0 in basis_0.qubits_group
            for qubits_1 in basis_1.qubits_group
        ]
    )
    return new_observed_basis


def combine(observable_0: Observable, observable_1: Observable) -> Observable:
    """二つの観測量から合成系の観測量を作る"""
    new_observed_values = [
        element_0["value"] * element_1["value"]
        for element_0 in observable_0.elements
        for element_1 in observable_1.elements
    ]
    new_observed_basis = combine_basis(
        observable_0.observed_basis, observable_1.observed_basis
    )

    new_observable = Observable(new_observed_values, new_observed_basis)
    return new_observable
