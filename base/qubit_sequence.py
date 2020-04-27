import numpy as np
from numpy.linalg import linalg

from base.conf import approx_digit
from base.error import InitializeError, InvalidSequenceError


class QubitSequence:
    """複数のQubitから成る、Qubitの列"""

    def __init__(self, bit_count: int, amplitudes: [complex]):
        # 確率振幅の要素数をチェック
        if len(amplitudes) != 2 ** bit_count:
            raise InvalidSequenceError("[ERROR]: Qubit数に対して、確率振幅の要素数が不正です")

        # 確率の総和をチェック
        if round(linalg.norm(amplitudes) - 1.0, approx_digit) != 0.0:
            raise InitializeError("[ERROR]: 確率の総和が1ではありません")

        # 初期化
        self.bit_count = bit_count
        self.amplitudes = np.array(amplitudes, complex)

    def __str__(self):
        term = ""
        for index in range(len(self.amplitudes)):
            vec_repl = format(index, "b").zfill(self.bit_count)
            term += f"{self.amplitudes[index]}|{vec_repl}>"

            # 最後以外はプラスと改行をつける
            if index != len(self.amplitudes) - 1:
                term += " +\n"

        return term
