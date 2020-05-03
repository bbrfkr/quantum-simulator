from math import sqrt

import pytest

from quantum_simulator.base.pure_qubits import PureQubits


@pytest.fixture(
    params=[
        {
            "probabilities": [1.0],
            "pure_qubits": [PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j])],
        }
    ]
)
def valid_qubit_list(request):
    """妥当な単一qubitに対する、確率リストとPureQubitsリスト"""
    return request.param
