from math import sqrt

import pytest

from quantum_simulator.base.observable import ObservedBasis
from quantum_simulator.base.pure_qubits import PureQubits
from quantum_simulator.base.transformer import UnitaryTransformer


@pytest.fixture()
def valid_observed_basis_for_unitary(multi_particles_observed_basis):
    """複数Qubit系に対する妥当な観測基底の組のfixture"""
    return [multi_particles_observed_basis, multi_particles_observed_basis]


@pytest.fixture(
    params=[
        [
            [
                PureQubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                PureQubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                PureQubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                PureQubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
                PureQubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
                PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
                PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
                PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
            ],
            [
                PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            ],
        ]
    ]
)
def invalid_observed_basis_for_unitary(request):
    """複数Qubit系に対するQubit数の異なる観測基底の組のfixture"""
    return [ObservedBasis(request.param[0]), ObservedBasis(request.param[1])]


@pytest.fixture(
    params=[
        {
            "unitary": UnitaryTransformer(
                ObservedBasis([PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])]),
                ObservedBasis([PureQubits([0j, 1 + 0j]), PureQubits([1 + 0j, 0j])]),
            ),
            "target": PureQubits([1 + 0j, 0j]),
            "expected_qubits": PureQubits([0j, 1 + 0j]),
        },
        {
            "unitary": UnitaryTransformer(
                ObservedBasis([PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])]),
                ObservedBasis(
                    [
                        PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "target": PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
            "expected_qubits": PureQubits([1 + 0j, 0j]),
        },
        {
            "unitary": UnitaryTransformer(
                ObservedBasis(
                    [
                        PureQubits([[1 + 0j, 0j], [0j, 0j]]),
                        PureQubits([[0j, 1 + 0j], [0j, 0j]]),
                        PureQubits([[0j, 0j], [1 + 0j, 0j]]),
                        PureQubits([[0j, 0j], [0j + 1, 0j]]),
                    ]
                ),
                ObservedBasis(
                    [
                        PureQubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
                        PureQubits([[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]]),
                        PureQubits([[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]]),
                        PureQubits([[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]]),
                    ]
                ),
            ),
            "target": PureQubits(
                [[sqrt(0.25) + 0j, sqrt(0.25) + 0j], [sqrt(0.25) + 0j, sqrt(0.25) + 0j]]
            ),
            "expected_qubits": PureQubits([[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]]),
        },
    ]
)
def dict_for_test_operation_of_unitary(request):
    """ユニタリ変換のオペレーションテストのためのfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "unitaries": [
                UnitaryTransformer(
                    ObservedBasis([PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])]),
                    ObservedBasis([PureQubits([0j, 1 + 0j]), PureQubits([1 + 0j, 0j])]),
                ),
                UnitaryTransformer(
                    ObservedBasis(
                        [
                            PureQubits([[1 + 0j, 0j], [0j, 0j]]),
                            PureQubits([[0j, 1 + 0j], [0j, 0j]]),
                            PureQubits([[0j, 0j], [1 + 0j, 0j]]),
                            PureQubits([[0j, 0j], [0j, 1 + 0j]]),
                        ]
                    ),
                    ObservedBasis(
                        [
                            PureQubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
                            PureQubits([[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]]),
                            PureQubits([[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]]),
                            PureQubits([[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]]),
                        ]
                    ),
                ),
            ],
            "expected_unitary": UnitaryTransformer(
                ObservedBasis(
                    [
                        PureQubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        PureQubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        PureQubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        PureQubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
                        PureQubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
                        PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
                        PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
                        PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
                    ]
                ),
                ObservedBasis(
                    [
                        PureQubits(
                            [
                                [[0j, 0j], [0j, 0j]],
                                [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[0j, 0j], [0j, 0j]],
                                [[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[0j, 0j], [0j, 0j]],
                                [[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[0j, 0j], [0j, 0j]],
                                [[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]],
                                [[0j, 0j], [0j, 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]],
                                [[0j, 0j], [0j, 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]],
                                [[0j, 0j], [0j, 0j]],
                            ]
                        ),
                        PureQubits(
                            [
                                [[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]],
                                [[0j, 0j], [0j, 0j]],
                            ]
                        ),
                    ]
                ),
            ),
        }
    ]
)
def dict_for_test_combined_of_unitaries(request):
    """ユニタリ変換の結合のテストのためのfixture"""
    return request.param
