import pytest

from quantum_simulator.base.qubits import Qubits
from quantum_simulator.channel.finalizer import Finalizer
from quantum_simulator.channel.registers import Registers
from quantum_simulator.channel.state import State


@pytest.fixture(
    params=[
        {
            "finalizer": Finalizer(
                [0, 1],
                State(
                    Qubits(
                        [
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 1.0 + 0j, 0j],
                            [0j, 0j, 0j, 0j],
                        ]
                    ),
                    Registers(2),
                ),
            ),
            "outcome": 2,
            "random_seed": 0,
        },
        {
            "finalizer": Finalizer(
                [0, 1],
                State(
                    Qubits(
                        [
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 1.0 + 0j],
                        ]
                    ),
                    Registers(2),
                ),
            ),
            "outcome": 3,
            "random_seed": 0,
        },
        {
            "finalizer": Finalizer(
                [1],
                State(
                    Qubits(
                        [
                            [0j, 0j, 0j, 0j],
                            [0j, 1.0 + 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                        ]
                    ),
                    Registers(2),
                ),
            ),
            "outcome": 0,
            "random_seed": 0,
        },
        {
            "finalizer": Finalizer(
                [1],
                State(
                    Qubits(
                        [
                            [1.0 + 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j],
                        ]
                    ),
                    Registers(2),
                ),
            ),
            "outcome": 0,
            "random_seed": 0,
        },
        {
            "finalizer": Finalizer(
                [2, 0],
                State(
                    Qubits(
                        [
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        ]
                    ),
                    Registers(2),
                ),
            ),
            "outcome": 3,
            "random_seed": 0,
        },
        {
            "finalizer": Finalizer(
                [2, 0],
                State(
                    Qubits(
                        [
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j],
                            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        ]
                    ),
                    Registers(2),
                ),
            ),
            "outcome": 2,
            "random_seed": 0,
        },
    ]
)
def dict_for_test_success_finalizer_finalize(request):
    """finalizeメソッドテスト用のfixture"""
    return request.param
