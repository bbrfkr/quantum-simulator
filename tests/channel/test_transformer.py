import random

from quantum_simulator.base.utils import allclose
from quantum_simulator.channel.transformer import (
    ObserveTransformer,
    TimeEvolveTransformer,
)


class TestObserveTransformer:
    """
    ObserveTransformerクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_observe_transformer_transform(
        self, dict_for_test_success_observe_transformer_transform
    ):
        """transformメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_observe_transformer_transform
        random.seed(dict_for_test["random_seed"])
        qubits = dict_for_test["qubits"]
        registers = dict_for_test["registers"]
        state = dict_for_test["state"]
        observable = dict_for_test["observable"]
        register_index = dict_for_test["register_index"]
        transformer = ObserveTransformer(state, observable)
        result = transformer.transform(register_index)
        assert allclose(result.qubits.matrix, qubits.matrix)
        assert result.registers.get(register_index) == registers[register_index]


class TestTimeEvolveTransformer:
    """
    TimeEvolveTransformerクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_time_evolve_transformer_transform(
        self, dict_for_test_success_time_evolve_transformer_transform
    ):
        """transformメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_time_evolve_transformer_transform
        qubits = dict_for_test["qubits"]
        state = dict_for_test["state"]
        time_evolution = dict_for_test["time_evolution"]
        transformer = TimeEvolveTransformer(state, time_evolution)
        result = transformer.transform()
        assert allclose(result.qubits.matrix, qubits.matrix)
