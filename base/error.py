class BaseError(Exception):
    """基底例外クラス"""


# Qubit群に対するエラー
class InitializeError(BaseError):
    """初期化エラー"""


class QubitCountNotMatchError(BaseError):
    """演算するQubit群同士のQubit数が一致しないエラー"""


class NoQubitsInput(BaseError):
    """Qubit群の未入力時エラー"""


# 観測量に対するエラー
class NonOrthogonalError(BaseError):
    """非直交エラー"""


class CannotDistinguishError(BaseError):
    """状態識別不能エラー"""


# 量子回路に対するエラー
class IncompatibleDimensionError(BaseError):
    """操作時次元不整合エラー"""
