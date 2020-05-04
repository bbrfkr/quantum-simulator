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
            "qubit_count": 1,
            "matrix_dim": 2,
        },
        {
            "probabilities": [1.0],
            "pure_qubits": [PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j])],
            "qubit_count": 1,
            "matrix_dim": 2,
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
            "qubit_count": 3,
            "matrix_dim": 8,
        },
        {
            "probabilities": [0.4, 0.6],
            "pure_qubits": [
                PureQubits([[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]]),
                PureQubits([[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]]),
            ],
            "qubit_count": 2,
            "matrix_dim": 4,
        },
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
            "qubit_count": 3,
            "matrix_dim": 8,
        },
        {
            "probabilities": [0.1, 0.2, 0.3, 0.4],
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
                    [
                        [[0j, 0j], [0j, 0j]],
                        [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]],
                    ]
                ),
                PureQubits(
                    [
                        [[0j, 0j], [0j, 0j]],
                        [[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]],
                    ]
                ),
            ],
            "qubit_count": 3,
            "matrix_dim": 8,
        },
    ]
)
def valid_orthogonal_qubits_list(request):
    """妥当な単一qubitに対する、確率リストと直交しているが縮退しているPureQubitsリスト"""
    return request.param


normalize_factor_0 = 1 / sqrt(4 + 2 * sqrt(2))
normalize_factor_1 = 1 / sqrt(4 - 2 * sqrt(2))
eigen_value_0 = (2 + sqrt(2)) / 4
eigen_value_1 = (2 - sqrt(2)) / 4
eigen_state_0 = [(1 + sqrt(2)) * normalize_factor_0 + 0j, normalize_factor_0 + 0j]
eigen_state_1 = [(1 - sqrt(2)) * normalize_factor_1 + 0j, normalize_factor_1 + 0j]


@pytest.fixture(
    params=[
        {
            "probabilities": [0.5, 0.5],
            "pure_qubits": [
                PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([1 + 0j, 0j]),
            ],
            "expected_values": [eigen_value_0, eigen_value_1],
            "expected_states": [PureQubits(eigen_state_0), PureQubits(eigen_state_1)],
            "qubit_count": 1,
            "matrix_dim": 2,
        },
        {
            "probabilities": [0.5, 0.5],
            "pure_qubits": [
                PureQubits([[sqrt(0.5) + 0j, 0j], [sqrt(0.5) + 0j, 0j]]),
                PureQubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
            ],
            "expected_values": [0.25, 0.75],
            "expected_states": [
                PureQubits([[0j, 0j], [-sqrt(0.5) + 0j, sqrt(0.5) + 0j]]),
                PureQubits(
                    [[sqrt(2 / 3) + 0j, 0j], [sqrt(1 / 6) + 0j, sqrt(1 / 6) + 0j]]
                ),
            ],
            "qubit_count": 2,
            "matrix_dim": 4,
        },
    ]
)
def valid_non_orthogonal_qubits_list(request):
    """妥当な単一qubitに対する、確率リストと非直交なPureQubitsリスト"""
    return request.param
