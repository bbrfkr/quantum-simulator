import pytest

from quantum_simulator.channel.state import State
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.channel.registers import Registers

@pytest.fixture(
    params=[
        {
            "qubit_count": 2,
            "register_count": 2,
            "noise": None,
            "input": 3,
            "state": State(
                Qubits([
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 1.0 + 0j]
                ]),
                Registers(2)
            )
        }
    ]
)
def dict_for_test_success_channel_initialize(request):
    """initializeメソッドテスト用のfixture"""
    return request.param