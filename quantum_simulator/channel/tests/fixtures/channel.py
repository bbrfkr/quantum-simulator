import pytest

from ....base import observable, time_evolution
from ....base.qubits import Qubits
from ....major.observable import IDENT_OBSERVABLE, ZERO_PROJECTION
from ....major.time_evolution import IDENT_EVOLUTION, NOT_GATE
from ...registers import Registers
from ...state import State
from ...transformer import ObserveTransformer, TimeEvolveTransformer


@pytest.fixture(
    params=[
        {
            "qubit_count": 2,
            "register_count": 2,
            "initializers": [],
            "state": State(
                Qubits(
                    [
                        [1 + 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                    ]
                ),
                Registers(2),
            ),
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "state": State(
                Qubits(
                    [
                        [1 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    ]
                ),
                Registers(2),
            ),
        },
    ]
)
def dict_for_test_success_channel_initialize(request):
    """initializeメソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 1, 2],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 1],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [1, 2],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 2],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [1],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [2],
            "outcome": 0,
        },
        {
            "qubit_count": 4,
            "register_count": 4,
            "initializers": [],
            "output_indices": [0, 1, 2, 3],
            "outcome": 0,
        },
        {
            "qubit_count": 4,
            "register_count": 4,
            "initializers": [],
            "output_indices": [0, 2, 3],
            "outcome": 0,
        },
    ]
)
def dict_for_test_success_channel_finalize(request):
    """finalizeメソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 1, 2],
            "transformer": TimeEvolveTransformer(
                time_evolution.multiple_combine(
                    [NOT_GATE, IDENT_EVOLUTION, IDENT_EVOLUTION]
                )
            ),
            "index": None,
            "outcome": 4,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 1, 2],
            "transformer": ObserveTransformer(
                observable.multiple_combine(
                    [IDENT_OBSERVABLE, IDENT_OBSERVABLE, ZERO_PROJECTION]
                )
            ),
            "index": 1,
            "outcome": 0,
            "register": 1.0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 1, 2],
            "transformer": ObserveTransformer(
                observable.multiple_combine(
                    [IDENT_OBSERVABLE, ZERO_PROJECTION, IDENT_OBSERVABLE]
                )
            ),
            "index": 1,
            "outcome": 0,
            "register": 1.0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "initializers": [],
            "output_indices": [0, 1, 2],
            "transformer": ObserveTransformer(
                observable.multiple_combine(
                    [ZERO_PROJECTION, IDENT_OBSERVABLE, IDENT_OBSERVABLE]
                )
            ),
            "index": 1,
            "outcome": 0,
            "register": 1.0,
        },
    ]
)
def dict_for_test_success_channel_transform(request):
    """transformメソッドテスト用のfixture"""
    return request.param
