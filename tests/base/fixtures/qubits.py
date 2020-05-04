from math import sqrt

import pytest

from quantum_simulator.base.pure_qubits import PureQubits


@pytest.fixture(
    params=[
        {
            "probabilities": [0.3, 0.7],
            "pure_qubits": [
                PureQubits([1.0 + 0j, 0.0 + 0j]),
                PureQubits([0.0 + 0j, 1.0 + 0j]),
            ],
        },
        {
            "probabilities": [0.4, 0.6],
            "pure_qubits": [
                PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            ],
        },
        {
            "probabilities": [0.01, 0.03, 0.06, 0.12, 0.15, 0.18, 0.21, 0.24],
            "pure_qubits": [
                PureQubits(
                    [[[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
                ),
                PureQubits(
                    [
                        [[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]],
                        [[0j, 0j], [0j, 0j]],
                    ]
                ),
                PureQubits(
                    [[[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
                ),
                PureQubits(
                    [
                        [[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]],
                        [[0j, 0j], [0j, 0j]],
                    ]
                ),
                PureQubits(
                    [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]]
                ),
                PureQubits(
                    [
                        [[0j, 0j], [0j, 0j]],
                        [[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]],
                    ]
                ),
                PureQubits(
                    [[[0j, 0j], [0j, 0j]], [[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]]]
                ),
                PureQubits(
                    [
                        [[0j, 0j], [0j, 0j]],
                        [[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]],
                    ]
                ),
            ],
        },
    ]
)
def valid_orthogonal_non_degrated_qubits_list(request):
    """妥当な単一qubitに対する、確率リストと直交しかつ非縮退なPureQubitsリスト"""
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [1.0],
            "pure_qubits": [PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j])],
        },
        {"probabilities": [1.0], "pure_qubits": [PureQubits([0j, 1 + 0j])]},
        {
            "probabilities": [0.7, 0.3],
            "pure_qubits": [
                PureQubits(
                    [[[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
                ),
                PureQubits(
                    [
                        [[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]],
                        [[0j, 0j], [0j, 0j]],
                    ]
                ),
            ],
        },
    ]
)
def valid_orthogonal_degrated_qubits_list(request):
    """妥当な単一qubitに対する、確率リストと直交しているが縮退しているPureQubitsリスト"""
    return request.param
