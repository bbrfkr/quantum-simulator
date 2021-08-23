import pytest

import quantum_simulator.base.observable as observable
import quantum_simulator.base.time_evolution as time_evolution
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.channel.registers import Registers
from quantum_simulator.channel.state import State
from quantum_simulator.channel.transformer import (
    ObserveTransformer,
    TimeEvolveTransformer,
)
from quantum_simulator.major.observable import (
    IDENT_OBSERVABLE,
    ZERO_PROJECTION,
)
from quantum_simulator.major.time_evolution import IDENT_EVOLUTION, NOT_GATE


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
