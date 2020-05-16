"""
QPUのレジスタを表現するクラス群
"""

class Registers:
    """
    QPUの古典レジスタのクラス

    Attributes:
        count (int): 内包するレジスタの数
        values (List[float]): レジスタの値リスト
    """
    def __init__(self, count: int):
        """
        Args:
            count (int): 初期化するレジスタの数
        """
        values = [None for index in range(count)]
        self.count = count
        self.values = values

    def get(self, index: int) -> float:
        """
        指定された番号のレジスタに格納された値を取得する

        Args:
            index (int): 取得したいレジスタの番号
        """
        return self.values[index]

    def put(self, index: int, value: float):
        """
        指定された番号のレジスタに値を格納する

        Args:
            index (int): 値を格納したいレジスタの番号
            value (float): 格納する値
        """
        self.values[index] = value
