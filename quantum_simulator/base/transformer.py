"""
状態変換を行うクラス群
"""

from typing import List

import numpy as np
from numpy import conjugate

from quantum_simulator.base.error import (
    IncompatibleDimensionError,
    InitializeError,
    NotCompleteError,
)
from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits, combine_ons
from quantum_simulator.base.qubits import Qubits, is_qubits_dim, resolve_arrays
from quantum_simulator.base.utils import allclose


class UnitaryTransformer:
    """
    ユニタリ変換による自然な状態変換のクラス
        ndarray: ndarray形式のユニタリ変換
        matrix: 行列形式のユニタリ変換
        matrix_dim: 行列形式における行列の次元
    """

    def __init__(self, unitary_array: list):
        """
        初期化
            unitary_array: ユニタリ変換のリスト表現。行列形式とndarray形式を許容する
        """

        tmp_array = np.array(unitary_array, dtype=complex)

        # 次元のチェック
        if not is_qubits_dim(tmp_array):
            message = "[ERROR]: 与えられたリストはQubit系上の作用素ではありません"
            raise InitializeError(message)

        # 行列表現とndarray表現の導出
        matrix, ndarray = resolve_arrays(tmp_array)
        matrix_dim = matrix.shape[0]
        # ユニタリ性のチェック
        hermite_matrix = np.dot(matrix, conjugate(matrix.T))
        if not allclose(hermite_matrix, np.identity(matrix_dim)):
            message = "[ERROR]: 与えられたリストはユニタリ変換ではありません"
            raise InitializeError(message)

        # 初期化
        self.ndarray = ndarray
        self.matrix = matrix
        self.matrix_dim = matrix_dim

    def __str__(self):
        """ユニタリ変換の二次元行列表現を出力"""
        return str(self.matrix)

    def print_ndarray(self):
        """ユニタリ変換のndarray表現を出力"""
        print(str(self.array))

    def operate(self, qubits: Qubits) -> Qubits:
        """ユニタリ変換によるQubit群の操作"""
        if self.matrix_dim != qubits.matrix_dim:
            message = "[ERROR]: 変換対象のQubit数が不正です"
            raise IncompatibleDimensionError(message)

        transformed_matrix = np.dot(
            np.dot(self.matrix, qubits.matrix), conjugate(self.matrix.T)
        )

        return Qubits(transformed_matrix)


def create_from_onb(
    pre_ons: OrthogonalSystem, post_ons: OrthogonalSystem
) -> UnitaryTransformer:
    """変換元基底と変換後基底を指定して、ユニタリ変換を作る"""

    # 指定されたONSが全てONBでなければエラー
    if not (pre_ons.is_onb() and post_ons.is_onb()):
        message = "[ERROR]: 正規直交基底を指定してください"
        raise NotCompleteError(message)

    # 観測基底を構成するQubit群の個数同士が一致していなければエラー
    len_pre_ons = len(pre_ons.qubits_list)
    if len_pre_ons != len(post_ons.qubits_list):
        message = "[ERROR]: 変換後のQubit数が変換前と異なります"
        raise InitializeError(message)

    # 変換のndarrayの生成
    elements_matrices = [
        np.multiply.outer(
            post_ons.qubits_list[index].vector,
            conjugate(pre_ons.qubits_list[index].vector),
        )
        for index in range(len_pre_ons)
    ]

    matrix = elements_matrices[-1]
    for index in range(len_pre_ons - 1):
        matrix = np.add(matrix, elements_matrices[index])

    return UnitaryTransformer(matrix)


def combine(
    unitary_0: UnitaryTransformer, unitary_1: UnitaryTransformer
) -> UnitaryTransformer:
    """二つのユニタリ変換から合成系のユニタリ変換を作る"""

    # 各ユニタリ行列を標準基底からの基底変換とみなして、ONBを抽出する
    matrix_0 = unitary_0.matrix
    matrix_0_dim = unitary_0.matrix_dim
    onb_0 = OrthogonalSystem(
        [
            PureQubits(list(conjugate(conjugate(matrix_0)[:, index])))
            for index in range(matrix_0_dim)
        ]
    )
    matrix_1 = unitary_1.matrix
    matrix_1_dim = unitary_1.matrix_dim
    onb_1 = OrthogonalSystem(
        [
            PureQubits(list(conjugate(conjugate(matrix_1)[:, index])))
            for index in range(matrix_1_dim)
        ]
    )

    # 合成系における変換後の基底を求める
    post_onb = combine_ons(onb_0, onb_1)

    # 合成系における変換前の基底である標準基底を求める
    identity = np.identity(matrix_0_dim * matrix_1_dim)
    pre_onb = OrthogonalSystem(
        [
            PureQubits(list(identity[:, index]))
            for index in range(matrix_0_dim * matrix_1_dim)
        ]
    )

    new_unitary = create_from_onb(pre_onb, post_onb)
    return new_unitary


def multiple_combine(unitaries: List[UnitaryTransformer]) -> UnitaryTransformer:
    """一般的に二つ以上のユニタリ変換から合成系のユニタリ変換を作る"""
    combined_unitary = unitaries[0]

    for index in range(len(unitaries) - 1):
        combined_unitary = combine(combined_unitary, unitaries[index + 1])

    return combined_unitary
