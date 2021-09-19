from math import sqrt

import pytest

from ....base.observable import Observable
from ....base.qubits import Qubits
from ....base.time_evolution import TimeEvolution
from ...registers import Registers
from ...state import State


@pytest.fixture(
    params=[
        {
            "state": State(Qubits([[0j, 0j], [0j, 1.0 + 0j]]), Registers(1)),
            "observable": Observable([[0j, 0j], [0j, 3.0 + 0j]]),
            "register_index": 0,
            "qubits": Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            "registers": [3.0],
            "random_seed": 0,
        },
        {
            "state": State(
                Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]), Registers(10)
            ),
            "observable": Observable([[0j, 0j], [0j, 3.0 + 0j]]),
            "register_index": 5,
            "qubits": Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            "registers": [None, None, None, None, None, 3.0, None, None, None, None],
            "random_seed": 0,
        },
    ]
)
def dict_for_test_success_observe_transformer_transform(request):
    """ObserveTransformerクラスのtransformメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "state": State(Qubits([[0j, 0j], [0j, 1.0 + 0j]]), Registers(1)),
            "time_evolution": TimeEvolution(
                [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]
            ),
            "qubits": Qubits([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]),
        },
        {
            "state": State(
                Qubits([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]), Registers(1)
            ),
            "time_evolution": TimeEvolution(
                [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]
            ),
            "qubits": Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
        },
    ]
)
def dict_for_test_success_time_evolve_transformer_transform(request):
    """TimeEvolveTransformerクラスのtransformメソッドに対する正常系テスト用fixture"""
    return request.param
