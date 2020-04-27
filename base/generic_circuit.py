import numpy as np
from numpy import identity
from numpy import matrix as mat
from numpy import ndarray, subtract

from base.conf import approx_digit
from base.error import IncompatibleDimensionError, InitializeError
from base.qubit_sequence import QubitSequence


class GenericCircuit:
    """一般的な量子回路のクラス"""

    def __init__(self, dim: int, matrix: [[complex]]):
        # 次元が2の階乗であるかチェックする
        rescue_dim = dim
        while dim != 1:
            if dim % 2 != 0:
                raise InitializeError("[ERROR]: この量子回路は次元が2^nではありません")
            dim /= 2
        dim = rescue_dim

        # 行列がユニタリ変換になっているかチェックする
        # まずnumpyで処理可能なようにndarray型にする
        rescue_matrix = matrix
        matrix = np.array(matrix)

        # 与えられた行列からhermite行列を作る
        matrix_star = mat.conjugate(matrix)
        hermite = mat.dot(matrix, matrix_star)

        # hermite行列が単位行列になるかチェック
        # 誤差を丸めるためにnumpy.roundを使うが、要素が一般的に複素数のため
        # 各要素の絶対値を求めてから丸める
        expected_zero = np.round(
            np.abs(hermite - identity(int(dim), complex)), approx_digit
        )
        if np.any(expected_zero != 0.0):
            raise InitializeError("[ERROR]: この量子回路はユニタリ変換ではありません")

        # 初期化
        self.dim = dim
        self.matrix = matrix

    def __str__(self):
        circuit_str = ""
        for index in range(len(self.matrix)):
            circuit_str += f"{self.matrix[index]}\n"

        return circuit_str

    def operate(self, qubits: QubitSequence):
        """Qubit列の操作"""
        if self.dim != len(qubits.amplitudes):
            raise IncompatibleDimensionError

        qubits.amplitudes = np.dot(self.matrix, qubits.amplitudes)
        return qubits
