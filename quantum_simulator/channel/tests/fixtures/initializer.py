from math import sqrt

import pytest

from ....base.qubits import Qubits


@pytest.fixture(
    params=[
        {
            "qubit_count": 1,
            "register_count": 2,
            "qubits": Qubits([[1.0 + 0j, 0j], [0j, 0j]]),
            "registers": [0.0, 0.0],
        },
        {
            "qubit_count": 3,
            "register_count": 5,
            "qubits": Qubits(
                [
                    [1 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                ]
            ),
            "registers": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
)
def dict_for_test_success_allocator_allocate(request):
    """allocateメソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "qubits": Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            "registers": [0.0, 0.0],
        },
        {
            "qubit_count": 1,
            "register_count": 2,
            "unitary": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "qubits": Qubits([[1 + 0j, 0j], [0j, 0j]]),
            "registers": [0.0, 0.0],
        },
    ]
)
def dict_for_test_success_initializer_initialize(request):
    """initializeメソッドテスト用のfixture"""
    return request.param
