from quantum_simulator.base.utils import is_pow2


class TestConf:
    """
    baseパッケージ汎用メソッド群のテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_false_is_pow2(self, not_pow2_int):
        result = is_pow2(not_pow2_int)
        assert result == False

    def test_for_true_is_pow2(self, pow2_int):
        result = is_pow2(pow2_int)
        assert result == True
