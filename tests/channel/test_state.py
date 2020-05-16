from quantum_simulator.channel.state import State


class TestState:
    """
    Stateクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_registers_constructor(self, dict_for_test_success_state_put):
        """コンストラクタの正常系テスト"""
        dict_for_test = dict_for_test_success_state_put
        for index in range(len(dict_for_test["values"])):
            dict_for_test["registers"].put(index, dict_for_test["values"][index])
        state = State(dict_for_test["qubits"], dict_for_test["registers"])

        assert state.qubits.qubit_count == dict_for_test["qubits"].qubit_count
        for index in range(len(dict_for_test["values"])):
            assert state.registers.get(index) == dict_for_test["values"][index]
