import pytest
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.channel.registers import Registers


@pytest.fixture(params=[
    {
        "qubits": Qubits([
            [0.7 + 0j, 0j, 0j, 0j],
            [0j, 0.3 + 0j, 0j, 0j],
            [0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j]
        ]),
        "registers": Registers(1),
        "values": [100.0]
    },
    {
        "qubits": Qubits([
            [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            [0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j],
            [0.5 + 0j, 0j, 0j, 0.5 + 0j]
        ]),
        "registers": Registers(3),
        "values": [100.0, -10.0, 1.0]
    }
])
def dict_for_test_success_state_put(request):
    """putメソッドの正常系テスト用fixture"""
    return request.param