from unittest.mock import patch

from quantum_simulator.base.utils import allclose
from quantum_simulator.channel.channel import Channel


class TestChannel:
    """
    Channelクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    @patch.multiple(Channel, __abstractmethods__=set())
    def test_for_success_channel_initialize(
        self, dict_for_test_success_channel_initialize
    ):
        """initializeメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_channel_initialize
        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        noise = dict_for_test["noise"]
        input = dict_for_test["input"]
        channel = Channel(qubit_count, register_count, noise)
        channel.initialize(input)

        state = channel.states[0]
        expected_state = dict_for_test["state"]
        assert allclose(state.qubits.matrix, expected_state.qubits.matrix)

    @patch.multiple(Channel, __abstractmethods__=set())
    def test_for_success_channel_finalize(self, dict_for_test_success_channel_finalize):
        """finalizeメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_channel_finalize
        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        noise = dict_for_test["noise"]
        input = dict_for_test["input"]
        channel = Channel(qubit_count, register_count, noise)
        channel.initialize(input)

        outcome = channel.finalize(dict_for_test["output_indices"])
        expected_outcome = dict_for_test["outcome"]
        assert outcome == expected_outcome

    @patch.multiple(Channel, __abstractmethods__=set())
    def test_for_success_channel_transform(
        self, dict_for_test_success_channel_transform
    ):
        """transformメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_channel_transform
        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        noise = dict_for_test["noise"]
        input = dict_for_test["input"]
        channel = Channel(qubit_count, register_count, noise)
        channel.initialize(input)

        channel.transform(dict_for_test["transformer"], dict_for_test["index"])
        outcome = channel.finalize(dict_for_test["output_indices"])
        expected_outcome = dict_for_test["outcome"]
        index = dict_for_test["index"]
        assert outcome == expected_outcome
        if index is not None:
            register = channel.states[-1].registers.get(index)
            expected_register = dict_for_test["register"]
            assert register == expected_register
