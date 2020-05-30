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
from quantum_simulator.major.observable import IDENT_OBSERVABLE, ONE_PROJECTION
from quantum_simulator.major.time_evolution import IDENT_EVOLUTION, NOT_GATE


@pytest.fixture(
    params=[
        {
            "qubit_count": 2,
            "register_count": 2,
            "noise": None,
            "input": 3,
            "state": State(
                Qubits(
                    [
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 1.0 + 0j],
                    ]
                ),
                Registers(2),
            ),
        },
        {
            "qubit_count": 2,
            "register_count": 2,
            "noise": None,
            "input": 2,
            "state": State(
                Qubits(
                    [
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 1.0 + 0j, 0j],
                        [0j, 0j, 0j, 0j],
                    ]
                ),
                Registers(2),
            ),
        },
        {
            "qubit_count": 2,
            "register_count": 2,
            "noise": None,
            "input": 1,
            "state": State(
                Qubits(
                    [
                        [0j, 0j, 0j, 0j],
                        [0j, 1.0 + 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                    ]
                ),
                Registers(2),
            ),
        },
        {
            "qubit_count": 2,
            "register_count": 2,
            "noise": None,
            "input": 0,
            "state": State(
                Qubits(
                    [
                        [1.0 + 0j, 0j, 0j, 0j],
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
            "noise": None,
            "input": 3,
            "state": State(
                Qubits(
                    [
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                    ]
                ),
                Registers(2),
            ),
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 1,
            "state": State(
                Qubits(
                    [
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
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
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "state": State(
                Qubits(
                    [
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j],
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
            "noise": None,
            "input": 6,
            "output_indices": [0, 1, 2],
            "outcome": 6,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [0, 1],
            "outcome": 2,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [1, 2],
            "outcome": 3,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [0, 2],
            "outcome": 2,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [0],
            "outcome": 0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [1],
            "outcome": 1,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [2],
            "outcome": 1,
        },
        {
            "qubit_count": 4,
            "register_count": 4,
            "noise": None,
            "input": 15,
            "output_indices": [0, 1, 2, 3],
            "outcome": 15,
        },
        {
            "qubit_count": 4,
            "register_count": 4,
            "noise": None,
            "input": 15,
            "output_indices": [0, 2, 3],
            "outcome": 7,
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
            "noise": None,
            "input": 6,
            "output_indices": [0, 1, 2],
            "transformer": TimeEvolveTransformer(
                time_evolution.multiple_combine(
                    [IDENT_EVOLUTION, IDENT_EVOLUTION, NOT_GATE]
                )
            ),
            "index": None,
            "outcome": 7,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [0, 1, 2],
            "transformer": ObserveTransformer(
                observable.multiple_combine(
                    [IDENT_OBSERVABLE, IDENT_OBSERVABLE, ONE_PROJECTION]
                )
            ),
            "index": 1,
            "outcome": 6,
            "register": 0.0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [0, 1, 2],
            "transformer": ObserveTransformer(
                observable.multiple_combine(
                    [IDENT_OBSERVABLE, ONE_PROJECTION, IDENT_OBSERVABLE]
                )
            ),
            "index": 1,
            "outcome": 6,
            "register": 1.0,
        },
        {
            "qubit_count": 3,
            "register_count": 3,
            "noise": None,
            "input": 6,
            "output_indices": [0, 1, 2],
            "transformer": ObserveTransformer(
                observable.multiple_combine(
                    [ONE_PROJECTION, IDENT_OBSERVABLE, IDENT_OBSERVABLE]
                )
            ),
            "index": 1,
            "outcome": 6,
            "register": 1.0,
        },
    ]
)
def dict_for_test_success_channel_transform(request):
    """transformメソッドテスト用のfixture"""
    return request.param
