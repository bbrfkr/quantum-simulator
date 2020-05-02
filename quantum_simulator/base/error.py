"""
baseパッケージ内のエラークラス
"""


class BaseError(Exception):
    """基底例外クラス"""


class InitializeError(BaseError):
    """初期化エラー"""


class QubitCountNotMatchError(BaseError):
    """演算するQubit群同士のQubit数が一致しないエラー"""


class NoQubitsInputError(BaseError):
    """Qubit群の未入力時エラー"""


class ReductionError(BaseError):
    """縮約時エラー"""


class IncompatibleDimensionError(BaseError):
    """操作時次元不整合エラー"""
