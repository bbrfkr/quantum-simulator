"""
channelパッケージ内のエラークラス群
"""


class BaseError(Exception):
    """基底例外クラス"""


class FinalizeError(BaseError):
    """finalize時エラー"""
