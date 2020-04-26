class BaseError(Exception):
    """基底例外クラス"""


class InitializeError(BaseError):
    """初期化エラー"""


class NonOrthogonalError(BaseError):
    """非直交エラー"""
