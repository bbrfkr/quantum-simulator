"""
観測量に関するクラス群
"""

from random import choices
from typing import List, Optional, Tuple, cast

import numpy

from .error import InitializeError, NotMatchCountError, NotMatchDimensionError
from .pure_qubits import OrthogonalSystem
from .qubits import Qubits, is_qubits_dim
from .switch_cupy import xp_factory
from .utils import allclose, count_bits, is_real_close

np = xp_factory()  # typing: numpy


class Observable:  # pylint: disable=too-few-public-methods
    """
    観測量のクラス

    Attributes:
        matrix (np.ndarray): 行列形式の観測量
    """

    def __init__(self, hermite_matrix: list):
        """
        Args:
            hermite_matrix (list): 観測量の候補となるリスト。行列形式とndarray形式を許容する
        """
        matrix = np.array(hermite_matrix)

        # 次元のチェック
        if not is_qubits_dim(matrix):
            message = "与えられたリストはQubit系上の作用素ではありません"
            raise InitializeError(message)

        # エルミート性のチェック
        if not allclose(matrix, np.conj(matrix.T)):
            message = "与えられたリストはエルミート行列ではありません"
            raise InitializeError(message)

        # 初期化
        self.matrix = matrix

    def __str__(self):
        """
        観測量の行列表現の文字列を返す

        Returns:
            str: 観測量の行列表現の文字列
        """
        return str(self.matrix)

    def expected_value(self, target: Qubits) -> float:
        """
        対象Qubitsに対する観測量の期待値を返す

        Args:
            target (Qbubits): 計算対象のQubits

        Returns:
            float: 観測量の期待値
        """

        # 観測量の対象空間内にQubitが存在するかチェック
        if target.qubit_count != (count_bits(self.matrix.shape[0]) - 1):
            message = "観測量の対象空間にQubit群が存在しません"
            raise NotMatchDimensionError(message)

        # 期待値の導出 trAρ
        expected_value = np.einsum("ij,ji", self.matrix, target.matrix)

        del target
        return expected_value


def _resolve_observed_results(
    eigen_values: List[float], eigen_states: numpy.ndarray
) -> Tuple[List[float], List[Observable]]:
    """
    与えられた固有値リストと固有ベクトルリストから、取りうる観測結果 (固有値と射影の組) を返す

    Args:
        eigen_values (List[float]): 固有値リスト
        eigen_states: (List[np.ndarray]): 固有ベクトルのリスト

    Returns:
        Tuple[List[float], List[Observable]]: 固有値と射影観測量の組
    """

    # 固有値の近似的に一意のリストと一致していたインデックスのリストを作る
    unique_eigen_values = []
    degrated_indice_list = []

    for index_0 in range(len(eigen_values)):
        # 既に走査した固有値はスキップ
        if not eigen_values[index_0] in unique_eigen_values:
            # 最初に自分自身をインデックスに登録する
            degrated_indice = [index_0]

            for index_1 in range(len(eigen_values) - index_0 - 1):
                if is_real_close(
                    eigen_values[index_0], eigen_values[index_0 + index_1 + 1]
                ):
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
        projection = np.einsum(
            "i,j", eigen_states[:, last_index], np.conj(eigen_states[:, last_index])
        )

        for index_1 in range(len(degrated_indice_list[index_0]) - 1):
            # インデックスリストからインデックスを取り出し
            # 射影行列同士を足して、目的の射影行列を作る
            target_index = degrated_indice_list[index_0][index_1]
            projection = np.add(
                projection,
                np.einsum(
                    "i,j",
                    eigen_states[:, target_index],
                    np.conj(eigen_states[:, target_index]),
                ),
            )

        projections.append(Observable(projection))

    del projection, last_index, eigen_values, eigen_states
    return (unique_eigen_values, projections)


def create_from_ons(observed_values: List[float], ons: OrthogonalSystem) -> Observable:
    """
    観測値リストと正規直交系から観測量を作る

    Args:
        observed_values (List[float]): 観測値のリスト
        ons (OrthogonalSystem): 正規直交系

    Returns:
        Observable: 導出された観測量
    """
    len_qubits_list = len(ons.qubits_list)

    # 観測値リストとONS内のPureQubitsリストの要素数同士が一致するかチェック
    if len_qubits_list != len(observed_values):
        message = "与えられた観測値リストと正規直交系を構成するQubitsリストの要素数が一致しません"
        raise NotMatchCountError(message)

    qubits_list = ons.qubits_list
    new_hermite_array = observed_values[-1] * np.outer(
        qubits_list[-1].vector, np.conj(qubits_list[-1].vector)
    )
    for index in range(len(observed_values) - 1):
        new_hermite_array = np.add(
            new_hermite_array,
            observed_values[index]
            * np.outer(qubits_list[index].vector, np.conj(qubits_list[index].vector)),
        )

    del observed_values, ons
    return Observable(new_hermite_array)


def observe(observable: Observable, target: Qubits) -> Tuple[float, Qubits]:
    """
    Qubit系に対して観測を実施し、観測値および収束後のQubitsを返す

    Args:
        observable (Observable): 使用する観測量
        target (Qubits): 観測対象のQubits

    Returns:
        Tuple[float, Qubits]: 観測値と収束後のQubits
    """

    # 観測の取りうる結果のリストを作る
    # まず近似的に一意な固有値リストと射影行列のリストを導出
    eigen_values, eigen_states = np.linalg.eigh(observable.matrix)
    observed_results_tuple = _resolve_observed_results(eigen_values, eigen_states)
    observed_results = [
        (observed_results_tuple[0][index], observed_results_tuple[1][index])
        for index in range(len(observed_results_tuple[0]))
    ]

    # 各観測結果に対する射影観測の期待値(観測確率)を求める
    observed_probabilities = [
        observed_results[index][1].expected_value(target)
        for index in range(len(observed_results))
    ]

    # 観測結果のランダムサンプリング
    observed_index = choices(
        range(len(observed_probabilities)), observed_probabilities
    )[0]
    observed_probability = observed_probabilities[observed_index]
    observed_result = observed_results[observed_index]
    del observed_results_tuple, observed_results, observed_probabilities

    # 観測によるQubitの収束 - 射影の適用と規格化
    projection_matrix = observed_result[1].matrix
    target_matrix = target.matrix

    # 射影行列を両側から挟み、かつトレース値で割って規格化する
    post_matrix = projection_matrix @ target_matrix @ projection_matrix
    normalized_post_matrix = (1.0 / observed_probability) * post_matrix
    del projection_matrix, target_matrix, post_matrix, observed_probability

    # 観測値の返却
    return (observed_result[0], Qubits(normalized_post_matrix))


def combine(observable_0: Optional[Observable], observable_1: Observable) -> Observable:
    """
    2つの観測量を結合して合成系の観測量を作る

    Args:
        observable_0 (Optional[Observable]): 結合される側の観測量
        observable_1 (Observable): 結合する側の観測量

    Returns:
        Observable: 結合後の観測量
    """
    if observable_0 is None:
        return observable_1

    # 新しい観測量の生成
    observable_0_matrix = list(observable_0.matrix)
    new_matrix = np.vstack(
        tuple(
            [
                np.hstack(
                    tuple(
                        [element * observable_1.matrix for element in observable_0_row]
                    )
                )
                for observable_0_row in observable_0_matrix
            ]
        )
    )
    return Observable(new_matrix)


def multiple_combine(observables: List[Observable]) -> Observable:
    """
    一般的に2つ以上のの観測量を結合して合成系の観測量を作る

    Args:
        observables (List[Observable]): 結合対象の観測量のリスト

    Returns:
        Observable: 結合後の観測量
    """
    if not observables:
        message = "空のリストが与えられました"
        raise NotMatchCountError(message)

    combined_observable = None
    for observable in observables:
        combined_observable = combine(combined_observable, observable)

    # リストは空ではないかつ、combineは必ず値を返すことが保証されているのでキャストする
    casted_observable = cast(Observable, combined_observable)
    return casted_observable
