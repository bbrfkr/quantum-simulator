import random


class TestFinalizer:
    """
    Finalizerクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_finalizer_finalize(
        self, dict_for_test_success_finalizer_finalize
    ):
        """finalizerメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_finalizer_finalize
        random.seed(dict_for_test["random_seed"])

        finalizer = dict_for_test["finalizer"]
        outcome = finalizer.finalize(dict_for_test["state"])[0]

        expected_outcome = dict_for_test["outcome"]
        assert outcome == expected_outcome
