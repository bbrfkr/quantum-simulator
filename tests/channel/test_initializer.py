from quantum_simulator.base.time_evolution import TimeEvolution
from quantum_simulator.base.utils import allclose
from quantum_simulator.channel.initializer import Allocator, Initializer
from quantum_simulator.channel.registers import Registers
from quantum_simulator.channel.transformer import TimeEvolveTransformer


class TestAllocator:
    """
    Allocatorクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_allocator_allocate(
        self, dict_for_test_success_allocator_allocate
    ):
        """allocateメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_allocator_allocate

        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        qubits = dict_for_test["qubits"]
        registers = Registers(register_count)
        for index in range(len(dict_for_test["registers"])):
            if dict_for_test["registers"][index] is not None:
                registers.put(index, dict_for_test["registers"][index])

        state = Allocator(qubit_count, register_count).allocate()
        assert allclose(qubits.matrix, state.qubits.matrix)
        for index in range(register_count):
            assert state.registers.get(index) == registers.get(index)


class TestInitializer:
    """
    Initializerクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_initializer_initialize(
        self, dict_for_test_success_initializer_initialize
    ):
        """initializeメソッドの正常系テスト"""
        dict_for_test = dict_for_test_success_initializer_initialize

        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        qubits = dict_for_test["qubits"]
        registers = Registers(register_count)
        for index in range(len(dict_for_test["registers"])):
            if dict_for_test["registers"][index] is not None:
                registers.put(index, dict_for_test["registers"][index])
        time_evolutions = [
            TimeEvolveTransformer(TimeEvolution(dict_for_test["unitary"]))
        ]

        allocator = Allocator(qubit_count, register_count)
        initializer = Initializer(allocator, time_evolutions)
        state = initializer.initialize()
        assert allclose(qubits.matrix, state.qubits.matrix)
        for index in range(register_count):
            assert state.registers.get(index) == registers.get(index)
