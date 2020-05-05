from quantum_simulator.base.utils import is_pow2, is_probabilities


class TestConf:
    """
    baseパッケージ汎用メソッド群のテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_true_is_pow2(self, pow2_int):
        """is_pow2メソッドのTrue値ケーステスト"""
        result = is_pow2(pow2_int)
        assert result is True

    def test_for_false_is_pow2(self, not_pow2_int):
        """is_pow2メソッドのFalse値ケーステスト"""
        result = is_pow2(not_pow2_int)
        assert result is False

    def test_for_true_is_probabilities(self, probabilities_list):
        """is_probabilitiesのTrue値ケーステスト"""
        result = is_probabilities(probabilities_list)
        assert result is True

    def test_for_false_is_probabilities(self, not_probabilities_list):
        """is_probabilitiesのFalse値ケーステスト"""
        result = is_probabilities(not_probabilities_list)
        assert result is False
