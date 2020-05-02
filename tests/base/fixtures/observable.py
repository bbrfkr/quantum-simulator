from math import sqrt

import pytest

from quantum_simulator.base.observable import Observable, ObservedBasis
from quantum_simulator.base.qubits import Qubits


@pytest.fixture()
def observed_basis(orthogonal_qubits):
    """単一Qubit系に対する観測基底のfixture"""
    return ObservedBasis(orthogonal_qubits)


@pytest.fixture(
    params=[[100.0, -100.0], [100.0, 50.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
)
def valid_observed_values(request):
    """単一Qubit系に対する妥当な観測値のfixture"""
    return request.param


@pytest.fixture(params=[[], [100.0, -100.0, 1.0]])
def invalid_observed_values(request):
    """単一Qubit系に対する不正な観測値のfixture"""
    return request.param


@pytest.fixture(
    params=[
        [
            Qubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
        ],
        [
            Qubits(
                [[[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]]
            ),
        ],
    ]
)
def multi_particles_observed_basis(request):
    """3粒子Qubit系に対する観測基底のfixture"""
    return ObservedBasis(request.param)


@pytest.fixture(
    params=[
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
def valid_multi_particles_observed_values(request):
    """3粒子Qubit系に対する妥当な観測値のfixture"""
    return request.param


@pytest.fixture(
    params=[[], [100.0, -100.0, 10.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
)
def invalid_multi_particles_observed_values(request):
    """3粒子Qubit系に対する不正な観測値のfixture"""
    return request.param


@pytest.fixture()
def observable(valid_observed_value, observe_basis):
    """単一Qubitに対する観測量のfixture"""
    return Observable(observe_basis, observe_basis)


@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [100.0, -100.0],
                ObservedBasis([Qubits([1 + 0j, 0j]), Qubits([0j, 1 + 0j])]),
            ),
            "qubit": Qubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
            "expected_value": 40.0,
        },
        {
            "observable": Observable(
                [100.0, 50.0],
                ObservedBasis([Qubits([1 + 0j, 0j]), Qubits([0j, 1 + 0j])]),
            ),
            "qubit": Qubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
            "expected_value": 85.0,
        },
        {
            "observable": Observable(
                [1.0, 0.0],
                ObservedBasis(
                    [
                        Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        Qubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "qubit": Qubits([0 + 0j, 1 + 0j]),
            "expected_value": 0.5,
        },
        {
            "observable": Observable(
                [2.0, 1.0],
                ObservedBasis(
                    [
                        Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        Qubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "qubit": Qubits([0 + 0j, 1 + 0j]),
            "expected_value": 1.5,
        },
    ]
)
def dict_for_test_expected_value(request):
    """単一Qubitに対する観測量、観測対象Qubit、期待値の組のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [100.0, -100.0],
                ObservedBasis(
                    [
                        Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        Qubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "qubit": Qubits([1 + 0j, 0j]),
            "randomize_seed": 1,
        },
        {
            "observable": Observable(
                [10.0, 0.0],
                ObservedBasis([Qubits([1 + 0j, 0j]), Qubits([0j, 1 + 0j]),]),
            ),
            "qubit": Qubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
            "randomize_seed": 1,
        },
    ]
)
def dict_for_test_observation(request):
    """単一Qubitに対する観測量、観測対象Qubitのfixture"""
    return request.param


@pytest.fixture()
def compound_observable(
    valid_multi_particles_observed_values, multi_particles_observe_basis
):
    """3粒子Qubit系に対する観測量のfixture"""
    return Observable(
        valid_multi_particles_observed_values, multi_particles_observe_basis
    )


@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ObservedBasis(
                    [
                        Qubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
                    ]
                ),
            ),
            "qubits": Qubits(
                [
                    [[0j, sqrt(0.25) + 0j], [0j, 0j]],
                    [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
                ]
            ),
            "expected_value": 5,
        },
    ]
)
def dict_for_test_expected_value_with_compound_observable(request):
    """3粒子Qubit系に対する観測量、観測対象Qubits、期待値の組のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ObservedBasis(
                    [
                        Qubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
                    ]
                ),
            ),
            "qubits": Qubits(
                [
                    [[sqrt(0.25) + 0j, 0j], [0j, 0j]],
                    [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
                ]
            ),
            "randomize_seed": 1,
        },
    ]
)
def dict_for_test_observation_with_compound_observable(request):
    """3粒子Qubit系Qubitに対する観測量、観測対象Qubits、ランダムシードのfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "observable_group": [
                Observable(
                    [1, 2, 3, 4],
                    ObservedBasis(
                        [
                            Qubits([[1 + 0j, 0j], [0j, 0j]]),
                            Qubits([[0j, 1 + 0j], [0j, 0j]]),
                            Qubits([[0j, 0j], [1 + 0j, 0j]]),
                            Qubits([[0j, 0j], [0j, 1 + 0j]]),
                        ]
                    ),
                ),
                Observable(
                    [100, -100, 1000, -1000],
                    ObservedBasis(
                        [
                            Qubits([[1 + 0j, 0j], [0j, 0j]]),
                            Qubits([[0j, 1 + 0j], [0j, 0j]]),
                            Qubits([[0j, 0j], [1 + 0j, 0j]]),
                            Qubits([[0j, 0j], [0j, 1 + 0j]]),
                        ]
                    ),
                ),
            ],
            "expected_matrix": [
                [100 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, -100 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 1000 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    -1000 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ],
                [0j, 0j, 0j, 0j, 200 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, -200 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 2000 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    -2000 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 300 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, -300 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 3000 + 0j, 0j, 0j, 0j, 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    -3000 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 400 + 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, -400 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 4000 + 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    -4000 + 0j,
                ],
            ],
        }
    ]
)
def dict_for_test_combine_observables(request):
    """2粒子Qubit系に対する観測量の組および結合後の観測量のfixture"""
    return request.param
