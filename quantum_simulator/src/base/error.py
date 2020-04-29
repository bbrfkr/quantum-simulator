class BaseError(Exception):
    """基底例外クラス"""


class InitializeError(BaseError):
    """初期化エラー"""


class QubitCountNotMatchError(BaseError):
    """演算するQubit群同士のQubit数が一致しないエラー"""


class NoQubitsInputError(BaseError):
    """Qubit群の未入力時エラー"""


# 量子回路に対するエラー
class IncompatibleDimensionError(BaseError):
    """操作時次元不整合エラー"""
