"""
観測量に関するクラス群
"""

from random import choices
from typing import List

import numpy as np

import quantum_simulator.base.pure_qubits as pure_qubits
from quantum_simulator.base.error import InitializeError, NotMatchDimensionError
from quantum_simulator.base.pure_qubits import PureQubits, combine
from quantum_simulator.base.qubits import (
    Qubits,
    is_qubits_dim,
    resolve_arrays,
    resolve_eigen,
)
from quantum_simulator.base.utils import is_real, isclose


class Observable:  # pylint: disable=too-few-public-methods
    """
    観測量のクラス
        eigen_values: 固有値のリスト
        eigen_states: 固有状態のリスト
        ndarray: ndarray形式の観測量
        matrix: 行列形式の観測量
        matrix_dim: 行列形式における行列の次元
    """

    def __init__(self, hermite_array: list):

        tmp_array = np.array(hermite_array, dtype=complex)

        # 次元のチェック
        if not is_qubits_dim(tmp_array):
            message = "[ERROR]: 与えられたリストはQubit系上の作用素ではありません"

        # 行列表現とndarray表現を導出
        matrix, ndarray = resolve_arrays(tmp_array)

        # 固有値、固有状態の導出
        eigen_values, eigen_states = resolve_eigen(matrix)

        # 固有値の虚部の有無をチェックし、floatに変換
        if not is_real(eigen_values):
            message = "[ERROR]: 与えられたリストには虚数の固有値が存在します"
            raise InitializeError(message)

        eigen_values = np.real(eigen_values)

        # 行列の次元を導出
        matrix_dim = matrix.shape[0]

        # 初期化
        self.eigen_values = eigen_values
        self.eigen_states = eigen_states
        self.ndarray = ndarray
        self.matrix = matrix
        self.matrix_dim = matrix_dim

    def __str__(self):
        """観測量の二次元行列表現を出力"""
        return str(self.matrix)

    def print_ndarray(self):
        """観測量のndarray表現を出力"""
        print(str(self.ndarray))

    def expected_value(self, target: Qubits) -> float:
        """対象Qubit群に対する観測量の期待値を返す"""

        # 観測量の対象空間内にQubitが存在するかチェック
        if target.qubit_count != self.eigen_states[0].qubit_count:
            message = "[ERROR]: 観測量の対象空間にQubit群が存在しません"
            raise NotMatchDimensionError(message)

        # 期待値の導出
        expected_value = np.trace(np.dot(self.matrix, target.matrix))
        return expected_value


def _resolve_observed_results(
    eigen_values: List[float], eigen_states: List[PureQubits]
) -> (List[float], List[Observable]):
    """固有値の縮退を検知し、観測に対して取りうる観測結果(固有値と射影行列の組)を返す"""

    # 固有値の近似的に一意のリストと一致していたインデックスのリストを作る
    unique_eigen_values = []
    degrated_indice_list = []

    for index_0 in range(len(eigen_values)):
        # 既に走査した固有値はスキップ
        if not eigen_values[index_0] in unique_eigen_values:
            # 最初に自分自身をインデックスに登録する
            degrated_indice = [index_0]

            for index_1 in range(len(eigen_values) - index_0 - 1):
                if isclose(eigen_values[index_0], eigen_values[index_0 + index_1 + 1]):
                    # 固有値が近似的に等しいときは、全ての固有値を一致させ、インデックスに登録
                    eigen_values[index_0 + index_1 + 1] = eigen_values[index_0]
                    degrated_indice.append(index_0 + index_1 + 1)

            unique_eigen_values.append(eigen_values[index_0])
            degrated_indice_list.append(degrated_indice)

    # リストから固有値に対応する射影作用素のリストを作る
    projections = []
    for index_0 in range(len(unique_eigen_values)):
        # 固有値に対応する射影作用素を作る
        # まず固有値が一致しているインデックスリストから
        # 最後のインデックスを取得し、対応する1次元射影行列を取り出す
        last_index = degrated_indice_list[index_0][-1]
        projection = eigen_states[last_index].projection_matrix

        for index_1 in range(len(degrated_indice_list[index_0]) - 1):
            # インデックスリストからインデックスを取り出し
            # 射影行列同士を足して、目的の射影行列を作る
            target_index = degrated_indice_list[index_0][index_1]
            projection = np.add(
                projection, eigen_states[target_index].projection_matrix
            )

        projections.append(Observable(projection))

    return (unique_eigen_values, projections)


def observe(observable: Observable, target: Qubits) -> float:
    """観測を実施して観測値を取得し、Qubits群を収束させる"""

    # 観測の取りうる結果のリストを作る
    # まず近似的に一意な固有値リストと射影行列のリストを導出
    observed_results_tuple = _resolve_observed_results(
        observable.eigen_values, observable.eigen_states
    )
    observed_results = [
        {
            "value": observed_results_tuple[0][index],
            "projection": observed_results_tuple[1][index],
        }
        for index in range(len(observed_results_tuple[0]))
    ]

    # 各観測結果に対する射影観測の期待値(観測確率)を求める
    observed_probabilities = [
        observed_results[index]["projection"].expected_value(target)
        for index in range(len(observed_results))
    ]

    # 観測結果のランダムサンプリング
    observed_result = choices(observed_results, observed_probabilities)[0]

    # 観測によるQubitの収束 - 射影の適用と規格化
    projection_matrix = observed_result["projection"].matrix
    target_matrix = target.matrix

    # 射影行列を両側から挟み、かつトレース値で割って規格化する
    post_matrix = np.dot(np.dot(projection_matrix, target_matrix), projection_matrix)
    observed_probability = np.trace(post_matrix)
    normalized_post_matrix = (1.0 / observed_probability) * post_matrix

    # 観測後の固有値、固有状態の再計算
    eigen_values, eigen_states = resolve_eigen(normalized_post_matrix)

    # Qubitsの収束
    target.eigen_values = eigen_values
    target.eigen_states = eigen_states
    target.matrix = normalized_post_matrix
    target.ndarray = normalized_post_matrix.reshape(target.ndarray.shape)

    # 観測値の返却
    return observed_result["value"]


def combine(observable_0: Observable, observable_1: Observable) -> Observable:
    """二つの観測量から合成系の観測量を作る"""

    # 固有値および固有状態を結合したリストを作成
    new_elements = [
        (observable_0.eigen_values[index_0] * observable_1.eigen_values[index_1])
        * pure_qubits.combine(
            observable_0.eigen_states[index_0], observable_1.eigen_states[index_1]
        ).projection_matrix
        for index_1 in range(len(observable_1.eigen_values))
        for index_0 in range(len(observable_0.eigen_values))
    ]

    new_hermite_array = new_elements[-1]
    for index in range(len(new_elements) - 1):
        new_hermite_array = np.add(new_hermite_array, new_elements[index])

    new_observable = Observable(new_hermite_array)
    return new_observable
