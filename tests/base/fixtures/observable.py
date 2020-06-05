from math import sqrt

import numpy
import pytest

from quantum_simulator.base.observable import Observable
from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits
from quantum_simulator.base.qubits import Qubits


@pytest.fixture(
    params=[
        # 単一Qubitに対する観測量
        {
            "target": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
            "matrix": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
            "ndarray": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
        },
        # 2粒子Qubitに対する観測量(行列形式)
        {
            "target": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "matrix": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "ndarray": [
                [[[1.0 + 0j, 0j], [0j, 0j]], [[0j, -2.0 + 0j], [0j, 0j]]],
                [[[0j, 0j], [3.0 + 0j, 0j]], [[0j, 0j], [0j, -4.0 + 0j]]],
            ],
        },
        # 2粒子Qubitに対する観測量(ndarray形式)
        {
            "target": [
                [[[1.0 + 0j, 0j], [0j, 0j]], [[0j, -2.0 + 0j], [0j, 0j]]],
                [[[0j, 0j], [3.0 + 0j, 0j]], [[0j, 0j], [0j, -4.0 + 0j]]],
            ],
            "matrix": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "ndarray": [
                [[[1.0 + 0j, 0j], [0j, 0j]], [[0j, -2.0 + 0j], [0j, 0j]]],
                [[[0j, 0j], [3.0 + 0j, 0j]], [[0j, 0j], [0j, -4.0 + 0j]]],
            ],
        },
        # 2粒子Qubitに対する観測量(行列形式、非標準基底固有状態)
        {
            "target": [
                [1.5 + 0j, -0.5 + 0j, 0j, 0j],
                [-0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, 3.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, 3.5 + 0j],
            ],
            "matrix": [
                [1.5 + 0j, -0.5 + 0j, 0j, 0j],
                [-0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, 3.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, 3.5 + 0j],
            ],
            "ndarray": [
                [[[1.5 + 0j, -0.5 + 0j], [0j, 0j]], [[-0.5 + 0j, 1.5 + 0j], [0j, 0j]]],
                [[[0j, 0j], [3.5 + 0j, -0.5 + 0j]], [[0j, 0j], [-0.5 + 0j, 3.5 + 0j]]],
            ],
        },
    ]
)
def test_for_success_observable_constructor(request):
    """__init__メソッドの正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubitに対する観測量
        {
            "observed_values": [1.0, -1.0],
            "ons": OrthogonalSystem(
                [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
            ),
            "matrix": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
            "ndarray": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
        },
        # 2粒子Qubitに対する観測量(行列形式)
        {
            "observed_values": [1.0, -2.0, 3.0, -4.0],
            "ons": OrthogonalSystem(
                [
                    PureQubits([1.0 + 0j, 0j, 0j, 0j]),
                    PureQubits([0j, 1.0 + 0j, 0j, 0j]),
                    PureQubits([0j, 0j, 1.0 + 0j, 0j]),
                    PureQubits([0j, 0j, 0j, 1.0 + 0j]),
                ]
            ),
            "matrix": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "ndarray": [
                [[[1.0 + 0j, 0j], [0j, 0j]], [[0j, -2.0 + 0j], [0j, 0j]]],
                [[[0j, 0j], [3.0 + 0j, 0j]], [[0j, 0j], [0j, -4.0 + 0j]]],
            ],
        },
        # 2粒子Qubitに対する観測量(行列形式、非標準基底固有状態)
        {
            "observed_values": [1.0, 2.0, 3.0, 4.0],
            "ons": OrthogonalSystem(
                [
                    PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j]),
                    PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j]),
                    PureQubits([0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                    PureQubits([0j, 0j, -sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                ]
            ),
            "matrix": [
                [1.5 + 0j, -0.5 + 0j, 0j, 0j],
                [-0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, 3.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, 3.5 + 0j],
            ],
            "ndarray": [
                [[[1.5 + 0j, -0.5 + 0j], [0j, 0j]], [[-0.5 + 0j, 1.5 + 0j], [0j, 0j]]],
                [[[0j, 0j], [3.5 + 0j, -0.5 + 0j]], [[0j, 0j], [-0.5 + 0j, 3.5 + 0j]]],
            ],
        },
    ]
)
def test_for_success_create_from_ons(request):
    """create_from_onsメソッドの正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 2粒子Qubitとこれに対する観測量
        {
            "observable": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "target": [
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0.5 + 0j],
                [0j, 0j, 0.5 + 0j, 0.5 + 0j],
            ],
            "expected_value": -0.5,
        },
        # 2粒子Qubitとこれに対する観測量
        {
            "observable": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 300.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "target": [
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0.5 + 0j],
                [0j, 0j, 0.5 + 0j, 0.5 + 0j],
            ],
            "expected_value": 148,
        },
    ]
)
def test_for_success_expected_value_for_pure(request):
    """expected_valueメソッドの純粋状態に対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit、縮退あり
        {
            "eigen_values": [2.0, 2.0],
            "eigen_states": [
                numpy.array([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                numpy.array([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            ],
            "expected_eigen_values": [2.0],
            "expected_projections": [Observable([[1.0 + 0j, 0j], [0j, 1.0 + 0j]])],
        },
        # 2粒子系、縮退あり
        {
            "eigen_values": [2.0, 2.0, -1.0, 1.0],
            "eigen_states": [
                numpy.array([1.0 + 0j, 0j, 0j, 0j]),
                numpy.array([0j, 1.0 + 0j, 0j, 0j]),
                numpy.array([0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                numpy.array([0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            ],
            "expected_eigen_values": [2.0, -1.0, 1.0],
            "expected_projections": [
                Observable(
                    [
                        [1.0 + 0j, 0j, 0j, 0j],
                        [0j, 1.0 + 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                    ]
                ),
                Observable(
                    [
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0.5 + 0j, 0.5 + 0j],
                        [0j, 0j, 0.5 + 0j, 0.5 + 0j],
                    ]
                ),
                Observable(
                    [
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0.5 + 0j, -0.5 + 0j],
                        [0j, 0j, -0.5 + 0j, 0.5 + 0j],
                    ]
                ),
            ],
        },
    ]
)
def test_for_success__resolve_observed_results(request):
    """_resolve_observed_resultsメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit、縮退あり
        {
            "observable": Observable([[1.0 + 0j, 0j], [0j, 2.0 + 0j]]),
            "target": Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            "expected_qubits": Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            "expected_observed_value": 2.0,
            "random_seed": 0,
        },
        {
            "observable": Observable([[1.0 + 0j, 0j], [0j, 2.0 + 0j]]),
            "target": Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            "expected_qubits": Qubits([[1.0 + 0j, 0j], [0j, 0j]]),
            "expected_observed_value": 1.0,
            "random_seed": 1,
        },
    ]
)
def test_for_success_observe(request):
    """observeメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit同士の結合
        {
            "target_0": Observable([[1.0 + 0j, 0j], [0j, -1.0 + 0j]]),
            "target_1": Observable([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            "matrix": [
                [0.5 + 0j, 0.5 + 0j, 0j, 0j],
                [0.5 + 0j, 0.5 + 0j, 0j, 0j],
                [0j, 0j, -0.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, -0.5 + 0j],
            ],
            "ndarray": [
                [[[0.5 + 0j, 0.5 + 0j], [0j, 0j]], [[0.5 + 0j, 0.5 + 0j], [0j, 0j]]],
                [
                    [[0j, 0j], [-0.5 + 0j, -0.5 + 0j]],
                    [[0j, 0j], [-0.5 + 0j, -0.5 + 0j]],
                ],
            ],
        },
        {
            "target_0": Observable([[1.0 + 0j, 0j], [0j, -1.0 + 0j]]),
            "target_1": Observable([[1.5 + 0j, 0.5 + 0j], [0.5 + 0j, 1.5 + 0j]]),
            "matrix": [
                [1.5 + 0j, 0.5 + 0j, 0j, 0j],
                [0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, -1.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, -1.5 + 0j],
            ],
            "ndarray": [
                [[[1.5 + 0j, 0.5 + 0j], [0j, 0j]], [[0.5 + 0j, 1.5 + 0j], [0j, 0j]]],
                [
                    [[0j, 0j], [-1.5 + 0j, -0.5 + 0j]],
                    [[0j, 0j], [-0.5 + 0j, -1.5 + 0j]],
                ],
            ],
        },
    ]
)
def test_for_success_observable_combine(request):
    """combineメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit同士の結合
        {
            "target_list": [
                Observable([[1.0 + 0j, 0j], [0j, 0j]]),
                Observable([[1.0 + 0j, 0j], [0j, -1.0 + 0j]]),
                Observable([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            ],
            "matrix": [
                [0.5 + 0j, 0.5 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0.5 + 0j, 0.5 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, -0.5 + 0j, -0.5 + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, -0.5 + 0j, -0.5 + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            ],
            "ndarray": [
                [
                    [
                        [[[0.5 + 0j, 0.5 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0.5 + 0j, 0.5 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                    [
                        [[[0j, 0j], [-0.5 + 0j, -0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [-0.5 + 0j, -0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                ],
                [
                    [
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                    [
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                ],
            ],
        },
    ]
)
def test_for_success_observable_multiple_combine(request):
    """multiple_combineメソッドに対する正常系テスト用fixture"""
    return request.param
