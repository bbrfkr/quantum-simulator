from math import sqrt

import pytest

from base.observable import Observable, ObserveBasis
from base.qubit import Qubit


# 妥当なqubitのfixture
@pytest.fixture(
    params=[
        [1 + 0j, 0j],
        [0j, 1 + 0j],
        [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
        [sqrt(0.5) * 1j, sqrt(0.5) * 1j],
        [sqrt(0.4) + 0j, sqrt(0.6) * 1j],
    ]
)
def valid_qubit_amp(request):
    return request.param


@pytest.fixture()
def valid_qubit(valid_qubit_amp):
    return Qubit(valid_qubit_amp[0], valid_qubit_amp[1])


# 不正なqubitのfixture
@pytest.fixture(
    params=[
        [0 + 0j, 0j],
        [sqrt(0.3) + 0j, sqrt(0.3) + 0j],
        [sqrt(0.3) * 1j, sqrt(0.3) * 1j],
        [sqrt(0.6) + 0j, sqrt(0.6) + 0j],
        [sqrt(0.6) * 1j, sqrt(0.6) * 1j],
    ]
)
def invalid_qubit_amp(request):
    return request.param


# 直交するqubit同士のfixture
@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [0j, 1 + 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, -sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.6) + 0j, -sqrt(0.4) * 1j]],
    ]
)
def orthogonal_qubits(request):
    qubits = [Qubit(amplitudes[0], amplitudes[1]) for amplitudes in request.param]
    return qubits


# 直交しないqubit同士のfixture
@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [1 + 0j, 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.4) + 0j, -sqrt(0.6) * 1j]],
    ]
)
def non_orthogonal_qubits(request):
    qubits = [Qubit(amplitudes[0], amplitudes[1]) for amplitudes in request.param]
    return qubits


# 観測基底のfixture
@pytest.fixture()
def observe_basis(orthogonal_qubits):
    return ObserveBasis(orthogonal_qubits[0], orthogonal_qubits[1])


# 妥当な観測値のfixture
@pytest.fixture(params=[[100.0, -100.0], [100.0, 50.0], [1.0, 0.0], [0.0, 1.0]])
def valid_observed_value(request):
    return request.param


# 不正な観測値のfixture
@pytest.fixture(params=[[0.0, 0.0], [100.0, 100.0], [1.0, 1.0], [-50.0, -50.0]])
def invalid_observed_value(request):
    return request.param


# 観測量のfixture
@pytest.fixture()
def observable(valid_observed_value, observe_basis):
    return Observable(valid_observed_value[0], valid_observed_value[1], observe_basis)


# 標準基底に対する観測量、観測対象Qubit、期待値の組
@pytest.fixture(
    params=[
        {
            "observable": Observable(
                100.0, -100.0, ObserveBasis(Qubit(1 + 0j, 0j), Qubit(0j, 1 + 0j))
            ),
            "qubit": Qubit(sqrt(0.7) + 0j, sqrt(0.3) + 0j),
            "expected_value": 40.0,
        },
        {
            "observable": Observable(
                100.0, 50.0, ObserveBasis(Qubit(1 + 0j, 0j), Qubit(0j, 1 + 0j))
            ),
            "qubit": Qubit(sqrt(0.7) + 0j, sqrt(0.3) + 0j),
            "expected_value": 85.0,
        },
        {
            "observable": Observable(
                1.0,
                0.0,
                ObserveBasis(
                    Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j),
                    Qubit(sqrt(0.5) + 0j, -sqrt(0.5) + 0j),
                ),
            ),
            "qubit": Qubit(0 + 0j, 1 + 0j),
            "expected_value": 0.5,
        },
        {
            "observable": Observable(
                2.0,
                1.0,
                ObserveBasis(
                    Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j),
                    Qubit(sqrt(0.5) + 0j, -sqrt(0.5) + 0j),
                ),
            ),
            "qubit": Qubit(0 + 0j, 1 + 0j),
            "expected_value": 1.5,
        },
    ]
)
def dict_for_test_expected_value(request):
    return request.param
