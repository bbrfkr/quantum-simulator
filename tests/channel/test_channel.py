from quantum_simulator.channel.channel import Channel
from quantum_simulator.base.utils import allclose

class TestChannel:
    """
    Channelクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

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


    # def test_for_success_channel_finalize(
    #     self, dict_for_test_success_channel_finalize
    # ):
    #     """finalizeメソッドの正常系テスト"""
    #     dict_for_test = dict_for_test_success_channel_finalize

    # def test_for_success_channel_transform(
    #     self, dict_for_test_success_channel_transform
    # ):
    #     """transformメソッドの正常系テスト"""
    #     dict_for_test = dict_for_test_success_channel_transform

    @patch.multiple(MyAbcClass, __abstractmethods__=set())
    def test(self):
         self.instance = MyAbcClass()
