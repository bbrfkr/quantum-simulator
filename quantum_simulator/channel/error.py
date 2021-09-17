"""
channelパッケージ内のエラークラス群
"""


class BaseError(Exception):
    """基底例外クラス"""


class AlreadyInitializedError(BaseError):
    """既にInitializeされていた時のエラー"""


class NotInitializedError(BaseError):
    """まだInitializeされていない時のエラー"""


class AlreadyFinalizedError(BaseError):
    """既にfinalizeされていた時のエラー"""
