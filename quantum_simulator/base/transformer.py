"""
状態変換を行うクラス群
"""

from math import sqrt

import numpy as np

from .error import IncompatibleDimensionError, InitializeError
from .observable import ObservedBasis, combine_basis
from .pure_qubits import PureQubits


class UnitaryTransformer:
    """ユニタリ変換による自然な状態変換のクラス"""

    def __init__(self, pre_basis: ObservedBasis, post_basis: ObservedBasis):
        # 観測基底を構成するQubit群の個数同士が一致していなければエラー
        len_pre_basis = len(pre_basis.qubits_group)
        if len_pre_basis != len(post_basis.qubits_group):
            message = "[ERROR]: 観測基底を構成するQubit群の個数同士が一致しません"
            raise InitializeError(message)

        self.pre_basis = pre_basis
        self.post_basis = post_basis

        # 変換のndarrayの生成
        elements_arrays = [
            np.multiply.outer(
                post_basis.qubits_group[index].amplitudes,
                np.conjugate(pre_basis.qubits_group[index].amplitudes),
            )
            for index in range(len_pre_basis)
        ]
        array = elements_arrays[-1]
        for index in range(len_pre_basis - 1):
            array = np.add(array, elements_arrays[index])

        self.array = array

        # 表現行列を導出する
        matrix_rank = int(sqrt(self.array.size))

        self.matrix_rank = matrix_rank
        self.matrix_shape = (self.matrix_rank, self.matrix_rank)
        self.matrix = self.array.reshape(self.matrix_shape)

    def __str__(self):
        """ユニタリ変換の二次元行列表現を出力"""
        return str(self.matrix)

    def print_array(self):
        """ユニタリ変換のndarray表現を出力"""
        print(str(self.array))

    def operate(self, qubits: PureQubits):
        """ユニタリ変換によるQubit群の操作"""
        if self.matrix_rank != qubits.amplitudes.size:
            message = "[ERROR]: 変換対象のQubit数が不正です"
            raise IncompatibleDimensionError(message)

        qubits.amplitudes = np.dot(self.matrix, qubits.vector).reshape(
            qubits.amplitudes.shape
        )


def combine(  # pylint: disable=C0330
    unitary_0: UnitaryTransformer, unitary_1: UnitaryTransformer
) -> UnitaryTransformer:
    """二つのユニタリ変換から合成系のユニタリ変換を作る"""

    new_pre_basis = combine_basis(unitary_0.pre_basis, unitary_1.pre_basis)
    new_post_basis = combine_basis(unitary_0.post_basis, unitary_1.post_basis)
    new_unitary = UnitaryTransformer(new_pre_basis, new_post_basis)
    return new_unitary
