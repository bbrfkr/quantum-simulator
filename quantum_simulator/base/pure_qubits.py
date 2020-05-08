"""
純粋状態のQubit系に関するクラス群
"""

from math import ceil
from typing import List, Tuple

import numpy as np
from numpy import conjugate

from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    QubitCountNotMatchError,
)
from quantum_simulator.base.utils import is_pow2, isclose


class PureQubits:
    """
    純粋状態で一般的に複数粒子のQubit系クラス

    Attributes:
        ndarray (numpy.array): ndarray形式のPureQubits
        vector (numpy.array): ベクトル形式のPureQubits
        qubit_count (int): PureQubitsに内包されているQubitの数
        projection (numpy.array): PureQubitsに対応する射影のndarray
        projection_matrix (numpy.array): PureQubitsに対応する射影行列
        projection_matrix_dim (int): 射影行列の次元
    """

    def __init__(self, amplitudes: list):
        """
        Args:
            amplitudes (list): 一般的に複素数の確率振幅のリスト。ベクトル形式とndarray形式を許容。
        """

        # Qubit系であるかチェック
        tmp_array = np.array(amplitudes, dtype=complex)
        if not _is_pure_qubits(tmp_array):
            message = "[ERROR]: 与えられたリストはQubit系に対応しません"
            raise InitializeError(message)

        # 各Qubit表現形式の導出
        vector, ndarray = _resolve_arrays(tmp_array)

        # 内包するQubit数を計算
        qubit_count = _count_qubits(ndarray)

        # 射影作用素を導出
        projection = np.multiply.outer(ndarray, np.conjugate(ndarray))

        # 射影作用素に対応する行列を導出
        projection_matrix_dim = 2 ** qubit_count
        projection_matrix = projection.reshape(
            projection_matrix_dim, projection_matrix_dim
        )

        # 初期化
        self.ndarray = ndarray
        self.vector = vector
        self.qubit_count = qubit_count
        self.projection = projection
        self.projection_matrix = projection_matrix
        self.projection_matrix_dim = projection_matrix_dim

    def __str__(self):
        """
        PureQubitsのベクトル表現を返す

        Returns:
            str: PureQubitsのベクトル表現
        """
        return str(self.vector)

    def print_ndarray(self):
        """
        PureQubitsのndarray表現を出力
        """
        print(self.ndarray)

    def print_projection_matrix(self):
        """
        PureQubitsの射影行列を出力
        """
        print(self.projection_matrix)

    def print_projection(self):
        """
        PureQubitsの射影行列に対応するndarray表現を出力
        """
        print(self.projection)

    def dirac_notation(self):
        """
        PureQubitsのDirac表記を出力
        """
        notation = ""
        vec_size = self.vector.size
        for index in range(vec_size):
            vec_repl = format(index, "b").zfill(self.qubit_count)
            notation += f"{self.vector[index]}|{vec_repl}>"

            # 最後以外はプラスと改行をつける
            if index != vec_size - 1:
                notation += " +\n"

        print(notation)


class OrthogonalSystem:
    """
    互いに直交する複数のPureQubits。正規直交系。

    Attributes:
        qubits_list (List[PureQubits]): 正規直交系を構成するPureQubitsのリスト
    """

    def __init__(self, qubits_list: List[PureQubits]):
        """
        Args:
            qubits_list (List[PureQubits]): 正規直交系を構成するPureQubitsのリスト
        """
        # 直交性の確認(相互にQubit数の確認も兼ねる)
        if not all_orthogonal(qubits_list):
            message = "[ERROR]: 与えられたQubit群のリストは互いに直交しません"
            raise InitializeError(message)

        self.qubits_list = qubits_list

    def is_onb(self):
        """
        正規直交系が正規直交基底であるか判定する。

        Returns:
            bool: 判定結果
        """
        # 基底を構成するQubit群の個数の確認
        if len(self.qubits_list) != self.qubits_list[0].vector.size:
            return False

        return True


def _is_pure_qubits(array: np.array) -> bool:
    """
    与えられたnumpy.arrayがQubit系を表現しているか判定する。

    Args:
        array (numpy.array): 判定対象のnumpy.array

    Returns:
        bool: 判定結果
    """

    # 要素数が2の累乗個であるかチェック
    size = array.size
    if not is_pow2(size):
        return False

    # ndarray形式の場合は、shapeの構成要素が全て2であるか
    # つまり、次元がQubitのテンソル積空間の次元かをチェック
    if len(array.shape) > 1:
        for sub_dim in array.shape:
            if sub_dim != 2:
                return False

    # 長さが1、つまり確率が1になるかをチェック
    norm = np.sqrt(np.sum(np.abs(array) ** 2))
    if not isclose(norm, 1.0):
        return False

    return True


def _count_qubits(pure_qubits: np.array) -> int:
    """
    与えられたnumpy.arrayがQubit系であることを仮定し、内包するQubit数を返す。

    Args:
        pure_qubits (numpy.array): PureQubitsの候補となるnumpy.array

    Returns:
        int: 内包するQubit数
    """
    size = pure_qubits.size
    count = 0
    while True:
        size /= 2
        count += 1
        if size % 2 == 1:
            break
    return count


def _resolve_arrays(pure_qubits: np.array) -> Tuple[np.array, np.array]:
    """
    与えられたnumpy.arrayがQubit系であることを仮定し、そのベクトル表現とndarray表現の組を返す。

    Args:
        pure_qubits (numpy.array): PureQubitsの候補となるnumpy.array

    Returns:
        Tuple[numpy.array, numpy.array]: pure_qubitsに対応する、ベクトル表現とndarray表現
    """
    vector = None
    ndarray = None

    if len(pure_qubits.shape) == 1:
        vector = pure_qubits
        qubit_count = _count_qubits(pure_qubits)
        ndarray_shape = tuple([2 for i in range(qubit_count)])
        ndarray = pure_qubits.reshape(ndarray_shape)
    else:
        ndarray = pure_qubits
        vector = pure_qubits.reshape(pure_qubits.size)

    return (vector, ndarray)


def combine(qubits_0: PureQubits, qubits_1: PureQubits) -> PureQubits:
    """
    二つのPureQubitsを結合し、その結果を返す。

    Args:
        qubits_0 (PureQubits): 結合される側のPureQubits
        qubits_1 (PureQubits): 結合する側のPureQubits

    Returns:
        PureQubits: 結合後のPureQubits。qubits_0 ⊗ qubits_1
    """
    new_ndarray = np.tensordot(qubits_0.ndarray, qubits_1.ndarray, 0)
    new_qubits = PureQubits(new_ndarray)

    return new_qubits


def combine_ons(ons_0: OrthogonalSystem, ons_1: OrthogonalSystem) -> OrthogonalSystem:
    """
    二つのOrthogonalSystemを要素順にを結合し、その結果を返す。

    Args:
        ons_0 (OrthogonalSystem): 結合される側のOrthogonalSystem
        ons_1 (OrthogonalSystem): 結合する側のOrthogonalSystem

    Returns:
        OrthogonalSystem: 結合後のOrthogonalSystem
    """
    new_qubits = [
        combine(qubits_0, qubits_1)
        for qubits_0 in ons_0.qubits_list
        for qubits_1 in ons_1.qubits_list
    ]

    new_ons = OrthogonalSystem(new_qubits)
    return new_ons


def multiple_combine(qubits_list: List[PureQubits]) -> PureQubits:
    """
    与えられたPureQubitsのリストを前方から順にを結合し、その結果を返す。

    Args:
        qubits_list (List[PureQubits]): 結合対象のPureQubitsのリスト

    Returns:
        PureQubits: 結合後のPureQubits。qubits_list[0] ⊗ ... ⊗ qubits_list[n]
    """
    combined_qubits = qubits_list[0]
    for index in range(len(qubits_list) - 1):
        combined_qubits = combine(combined_qubits, qubits_list[index + 1])

    return combined_qubits


def multiple_combine_ons(ons_list: List[OrthogonalSystem]) -> OrthogonalSystem:
    """
    与えられたOrthogonalSystemのリストを前方から順にを結合し、その結果を返す。

    Args:
        ons_lit (List[OrthogonalSystem]): 結合対象のOrthogonalSystemのリスト

    Returns:
        OrthogonalSystem: 結合後のOrthogonalSystem
    """
    combined_ons = ons_list[0]
    for index in range(len(ons_list) - 1):
        combined_ons = combine_ons(combined_ons, ons_list[index + 1])

    return combined_ons


def inner(qubits_0: PureQubits, qubits_1: PureQubits) -> complex:
    """
    PureQubits同士の内積を返す。

    Args:
        qubits_0 (PureQubits): ブラベクトルに対応するPureQubits。<qubits_0|
        qubits_1 (PureQubits): ケットベクトルに対応するPureQubits。|qubits_1>

    Returns:
        complex: qubits_0とqubits_1の内積。<qubits_0|qubits_1>
    """
    # 内積をとるQubit群同士のQubit数が一致してなければエラー
    if qubits_0.qubit_count != qubits_1.qubit_count:
        message = "[ERROR]: 対象PureQubits同士のQubit数が一致しません"
        raise QubitCountNotMatchError(message)

    return np.inner(conjugate(qubits_0.vector), qubits_1.vector)


def is_orthogonal(qubits_0: PureQubits, qubits_1: PureQubits) -> bool:
    """
    二つのPureQubits同士が直交しているかを判定する。

    Args:
        qubits_0 (PureQubits): 計算対象の1つめのPureQubits
        qubits_1 (PureQubits): 計算対象の2つめのPureQubits
    
    Returns:
        bool: qubits_0とqubits_1の内積が0か否か
    """
    return isclose(inner(qubits_0, qubits_1), 0.0)


def all_orthogonal(qubits_list: List[PureQubits]) -> bool:
    """
    複数のPureQubits同士が互いに直交しているかを判定する。

    Args:
        qubits_list (List[PureQubits]): 計算対象のPureQubitsのリスト

    Returns:
        bool: qubits_list内のPureQubits同士の内積が全て0か否か
    """

    len_qubits_list = len(qubits_list)

    # PureQubitsが一つも入力されない時はエラー
    if len_qubits_list == 0:
        message = "[ERROR]: 与えられたリストにPureQubitsが見つかりません"
        raise NoQubitsInputError(message)

    # PureQubitsが一つだけ与えられた時は明らかに互いに直交
    if len_qubits_list == 1:
        return True

    # PureQubitsが二つ以上与えられた場合
    for index_0 in range(ceil(len_qubits_list / 2)):
        for index_1 in range(len_qubits_list - index_0 - 1):
            if not is_orthogonal(
                qubits_list[index_0], qubits_list[len_qubits_list - index_1 - 1]
            ):
                return False

    return True
