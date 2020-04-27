from math import sqrt

import pytest

from base.qubit import Qubit
from base.observable import Observable, ObserveBasis

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
def valid_qubit(request):
    return request.param


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
def invalid_qubit(request):
    return request.param


# 直交するqubit同士のfixture
@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [0j, 1 + 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, - sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.6) + 0j, - sqrt(0.4) * 1j]],
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
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.4) + 0j, - sqrt(0.6) * 1j]],
    ]
)
def non_orthogonal_qubits(request):
    qubits = [Qubit(amplitudes[0], amplitudes[1]) for amplitudes in request.param]
    return qubits

# 標準観測基底のfixture
@pytest.fixture()
def standard_basis():
    return ObserveBasis(Qubit(1 + 0j, 0j), Qubit(0j, 1 + 0j))

# 標準射影観測量のfixture
@pytest.fixture()
def standard_projections(standard_basis):
    return [Observable(1.0, 0.0, standard_basis), Observable(0.0, 1.0, standard_basis)]

# 標準基底に対する観測量のfixture
@pytest.fixture(
    params=[
        [100.0,-100.0],
        [100.0,50.0]
    ]
)
def standard_observable(request, standard_basis):
    return Observable(request.param[0], request.param[1], standard_basis)