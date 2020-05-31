import sys
from unittest.mock import Mock

sys.modules["cupy"] = Mock()

import random

from quantum_simulator.base.time_evolution import TimeEvolution
from quantum_simulator.base.utils import allclose
from quantum_simulator.channel.initializer import Allocator, Initializer
from quantum_simulator.channel.registers import Registers


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
        random.seed(dict_for_test["random_seed"])

        input = dict_for_test["input"]
        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        qubits = dict_for_test["qubits"]
        registers = Registers(register_count)
        for index in range(len(dict_for_test["registers"])):
            if dict_for_test["registers"][index] is not None:
                registers.put(index, dict_for_test["registers"][index])

        state = Allocator(input, qubit_count, register_count).allocate()
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
        random.seed(dict_for_test["random_seed"])

        input = dict_for_test["input"]
        qubit_count = dict_for_test["qubit_count"]
        register_count = dict_for_test["register_count"]
        qubits = dict_for_test["qubits"]
        registers = Registers(register_count)
        for index in range(len(dict_for_test["registers"])):
            if dict_for_test["registers"][index] is not None:
                registers.put(index, dict_for_test["registers"][index])
        time_evolution = TimeEvolution(dict_for_test["unitary"])

        allocator = Allocator(input, qubit_count, register_count)
        initializer = Initializer(allocator, time_evolution)
        state = initializer.initialize()
        assert allclose(qubits.matrix, state.qubits.matrix)
        for index in range(register_count):
            assert state.registers.get(index) == registers.get(index)
