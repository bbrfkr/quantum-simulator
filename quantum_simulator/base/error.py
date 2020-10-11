"""
baseパッケージ内のエラークラス群
"""


class BaseError(Exception):
    """基底例外クラス"""


class InitializeError(BaseError):
    """インスタンスの初期化時エラー"""


class QubitCountNotMatchError(BaseError):
    """演算対象Qubit群同士のQubit数が一致しない場合のエラー"""


class NoQubitsInputError(BaseError):
    """演算対象Qubit群の未入力時エラー"""


class NotMatchCountError(BaseError):
    """リストの要素数が一致しない場合のエラー"""


class NotMatchDimensionError(BaseError):
    """２つの要素の空間次元数が一致しない場合のエラー"""


class InvalidProbabilitiesError(BaseError):
    """不正な確率リストが与えられた場合のエラー"""


class ReductionError(BaseError):
    """密度行列の不正縮約時エラー"""


class NotPureError(BaseError):
    """Qubitが純粋状態でない場合のエラー"""


class IncompatibleDimensionError(BaseError):
    """Qubit操作時の空間次元の不整合エラー"""


class NotCompleteError(BaseError):
    """正規直交系の不完全時エラー"""


class NegativeValueError(BaseError):
    """負数時のエラー"""


class OutOfRangeIndexError(BaseError):
    """不正なインデックスのエラー"""


class CombineError(BaseError):
    """Qubits合成時エラー"""
