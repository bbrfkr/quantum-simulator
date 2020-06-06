from quantum_simulator.channel.registers import Registers


class TestRegisters:
    """
    Registersクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_registers_constructor(self, length_registers):
        """コンストラクタの正常系テスト"""
        registers = Registers(length_registers)
        assert registers.count == length_registers
        assert len(registers.values) == length_registers

    def test_for_registers_put(self, dict_for_test_success_registers_put):
        """putメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_registers_put
        registers = Registers(dict_for_test["count"])
        registers.put(dict_for_test["insert_index"], dict_for_test["value"])
        for index in range(dict_for_test["count"]):
            if index == dict_for_test["insert_index"]:
                assert registers.values[index] == dict_for_test["value"]
            else:
                assert registers.values[index] == 0.0

    def test_for_registers_get(self, dict_for_test_success_registers_get):
        """getメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_registers_get
        registers = Registers(dict_for_test["count"])
        registers.put(dict_for_test["insert_index"], dict_for_test["value"])
        assert registers.get(dict_for_test["insert_index"]) == dict_for_test["value"]
