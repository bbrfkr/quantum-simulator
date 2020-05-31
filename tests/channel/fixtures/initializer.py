import sys
from unittest.mock import Mock

sys.modules["cupy"] = Mock()

from math import sqrt

import pytest

from quantum_simulator.base.qubits import Qubits


@pytest.fixture(
    params=[
        {
            "input": 1,
            "qubit_count": 1,
            "register_count": 2,
            "qubits": Qubits([[1.0 + 0j, 0j], [0j, 0j]]),
            "registers": [0.0, None],
            "random_seed": 1,
        },
        {
            "input": 2,
            "qubit_count": 3,
            "register_count": 5,
            "qubits": Qubits(
                [
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                ]
            ),
            "registers": [0.0, 1.0, 0.0, None, None],
            "random_seed": 4,
        },
    ]
)
def dict_for_test_success_allocator_allocate(request):
    """allocateメソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "input": 1,
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "qubits": Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            "registers": [0.0, None],
            "random_seed": 1,
        },
        {
            "input": 1,
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "qubits": Qubits([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]),
            "registers": [1.0, None],
            "random_seed": 0,
        },
        {
            "input": 1,
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "qubits": Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            "registers": [1.0, None],
            "random_seed": 0,
        },
        {
            "input": 0,
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "qubits": Qubits([[1.0 + 0j, 0j], [0j, 0j]]),
            "registers": [1.0, None],
            "random_seed": 0,
        },
        {
            "input": 1,
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "qubits": Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            "registers": [0.0, None],
            "random_seed": 1,
        },
        {
            "input": 0,
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "qubits": Qubits([[1.0 + 0j, 0j], [0j, 0j]]),
            "registers": [0.0, None],
            "random_seed": 1,
        },
    ]
)
def dict_for_test_success_initializer_initialize(request):
    """initializeメソッドテスト用のfixture"""
    return request.param
