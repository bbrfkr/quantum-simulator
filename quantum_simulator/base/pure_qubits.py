"""
純粋状態のQubit系に関するクラス群
"""

from math import ceil
from typing import List, Optional, cast

import numpy

from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    NotMatchCountError,
    QubitCountNotMatchError,
)
from quantum_simulator.base.switch_cupy import xp_factory
from quantum_simulator.base.utils import allclose, count_bits, is_pow2, isclose

np = xp_factory()  # typing: numpy


class PureQubits:
    """
    純粋状態で一般的に複数粒子のQubit系クラス

    Attributes:
        vector (np.ndarray): ベクトル形式のPureQubits
        qubit_count (int): PureQubitsに内包されているQubitの数
    """

    def __init__(self, amplitudes: list):
        """
        Args:
            amplitudes (list): 一般的に複素数の確率振幅のリスト。ベクトル形式とndarray形式を許容。
        """

        # Qubit系であるかチェック
        vector = np.ndarray(amplitudes)
        if not _is_pure_qubits(vector):
            message = "[ERROR]: 与えられたリストはQubit系に対応しません"
            raise InitializeError(message)

        # 内包するQubit数を計算
        qubit_count = count_bits(vector.size) - 1

        # 初期化
        self.vector = vector
        self.qubit_count = qubit_count
        del vector
        del qubit_count

    def __str__(self):
        """
        PureQubitsのベクトル表現を返す

        Returns:
            str: PureQubitsのベクトル表現
        """
        return str(self.vector)

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


def _is_pure_qubits(array: numpy.ndarray) -> bool:
    """
    与えられたnp.ndarrayがQubit系を表現しているか判定する。

    Args:
        array (np.ndarray): 判定対象のnp.ndarray

    Returns:
        bool: 判定結果
    """

    # 要素数が2の累乗個であるかチェック
    size = array.size
    if not is_pow2(size):
        return False

    # 長さが1、つまり確率が1になるかをチェック
    norm = np.sqrt(np.sum(np.abs(array) ** 2))
    if not allclose(norm, np.ndarray(1.0)):
        return False

    return True


def combine(qubits_0: Optional[PureQubits], qubits_1: PureQubits) -> PureQubits:
    """
    二つのPureQubitsを結合し、その結果を返す。

    Args:
        qubits_0 (Optional[PureQubits]): 結合される側のPureQubits
        qubits_1 (PureQubits): 結合する側のPureQubits

    Returns:
        PureQubits: 結合後のPureQubits。qubits_0 ⊗ qubits_1
    """
    if qubits_0 is None:
        return qubits_1

    qubits_0_vector = list(qubits_0.vector)
    new_vector = np.hstack(
        tuple([element * qubits_1.vector for element in qubits_0_vector])
    )
    new_qubits = PureQubits(new_vector)

    del new_vector, qubits_0_vector, qubits_0, qubits_1
    return new_qubits


def combine_ons(
    ons_0: Optional[OrthogonalSystem], ons_1: OrthogonalSystem
) -> OrthogonalSystem:
    """
    二つのOrthogonalSystemを要素順にを結合し、その結果を返す。

    Args:
        ons_0 (Optional[OrthogonalSystem]): 結合される側のOrthogonalSystem
        ons_1 (OrthogonalSystem): 結合する側のOrthogonalSystem

    Returns:
        OrthogonalSystem: 結合後のOrthogonalSystem
    """
    if ons_0 is None:
        return ons_1

    new_qubits = [
        combine(qubits_0, qubits_1)
        for qubits_0 in ons_0.qubits_list
        for qubits_1 in ons_1.qubits_list
    ]
    new_ons = OrthogonalSystem(new_qubits)

    del new_qubits, ons_0, ons_1
    return new_ons


def multiple_combine(qubits_list: List[PureQubits]) -> PureQubits:
    """
    与えられたPureQubitsのリストを前方から順にを結合し、その結果を返す。

    Args:
        qubits_list (List[PureQubits]): 結合対象のPureQubitsのリスト

    Returns:
        PureQubits: 結合後のPureQubits。qubits_list[0] ⊗ ... ⊗ qubits_list[n]
    """
    if not qubits_list:
        message = "[ERROR]: 空のリストが与えられました"
        raise NotMatchCountError(message)

    combined_qubits = None
    for qubits in qubits_list:
        combined_qubits = combine(combined_qubits, qubits)

    # リストは空ではないかつ、combineは必ず値を返すことが保証されているのでキャストする
    casted_qubits = cast(PureQubits, combined_qubits)
    return casted_qubits


def multiple_combine_ons(ons_list: List[OrthogonalSystem]) -> OrthogonalSystem:
    """
    与えられたOrthogonalSystemのリストを前方から順にを結合し、その結果を返す。

    Args:
        ons_list (List[OrthogonalSystem]): 結合対象のOrthogonalSystemのリスト

    Returns:
        OrthogonalSystem: 結合後のOrthogonalSystem
    """
    if not ons_list:
        message = "[ERROR]: 空のリストが与えられました"
        raise NotMatchCountError(message)

    combined_ons = None
    for ons in ons_list:
        combined_ons = combine_ons(combined_ons, ons)

    # リストは空ではないかつ、combineは必ず値を返すことが保証されているのでキャストする
    casted_ons = cast(OrthogonalSystem, combined_ons)
    return casted_ons


def inner(qubits_0: PureQubits, qubits_1: PureQubits) -> complex:
    """
    PureQubits同士の内積を返す。

    Args:
        qubits_0 (PureQubits): ブラベクトルに対応するPureQubits ＜qubits_0｜
        qubits_1 (PureQubits): ケットベクトルに対応するPureQubits。｜qubits_1＞

    Returns:
        complex: qubits_0とqubits_1の内積。＜qubits_0｜qubits_1＞
    """
    # 内積をとるQubit群同士のQubit数が一致してなければエラー
    if qubits_0.qubit_count != qubits_1.qubit_count:
        message = "[ERROR]: 対象PureQubits同士のQubit数が一致しません"
        raise QubitCountNotMatchError(message)

    return np.vdot(qubits_0.vector, qubits_1.vector)


def is_orthogonal(qubits_0: PureQubits, qubits_1: PureQubits) -> bool:
    """
    二つのPureQubits同士が直交しているかを判定する。

    Args:
        qubits_0 (PureQubits): 計算対象の1つめのPureQubits
        qubits_1 (PureQubits): 計算対象の2つめのPureQubits

    Returns:
        bool: qubits_0とqubits_1の内積が0か否か
    """
    return isclose(inner(qubits_0, qubits_1), 0.0 + 0j)


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
