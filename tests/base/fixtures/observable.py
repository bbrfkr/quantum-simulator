from math import sqrt

import pytest

from quantum_simulator.base.observable import Observable
from quantum_simulator.base.pure_qubits import PureQubits, OrthogonalSystem
from quantum_simulator.base.qubits import Qubits


@pytest.fixture(
    params=[
        # 単一Qubitに対する観測量
        {
            "target": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
            "eigen_values": [1.0, -1.0],
            "eigen_states": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "matrix": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
            "matrix_dim": 2,
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
            "eigen_values": [1.0, -2.0, 3.0, -4.0],
            "eigen_states": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 1.0 + 0j, 0j],
                [0j, 0j, 0j, 1.0 + 0j],
            ],
            "matrix": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "matrix_dim": 4,
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
            "eigen_values": [1.0, -2.0, 3.0, -4.0],
            "eigen_states": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 1.0 + 0j, 0j],
                [0j, 0j, 0j, 1.0 + 0j],
            ],
            "matrix": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "matrix_dim": 4,
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
            "eigen_values": [1.0, 2.0, 3.0, 4.0],
            "eigen_states": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [0j, 0j, -sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "matrix": [
                [1.5 + 0j, -0.5 + 0j, 0j, 0j],
                [-0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, 3.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, 3.5 + 0j],
            ],
            "matrix_dim": 4,
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
            "ons": OrthogonalSystem([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]),
            "eigen_values": [1.0, -1.0],
            "eigen_states": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "matrix": [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
            "matrix_dim": 2,
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
            "eigen_values": [1.0, -2.0, 3.0, -4.0],
            "eigen_states": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 1.0 + 0j, 0j],
                [0j, 0j, 0j, 1.0 + 0j],
            ],
            "matrix": [
                [1.0 + 0j, 0j, 0j, 0j],
                [0j, -2.0 + 0j, 0j, 0j],
                [0j, 0j, 3.0 + 0j, 0j],
                [0j, 0j, 0j, -4.0 + 0j],
            ],
            "matrix_dim": 4,
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
            "eigen_values": [1.0, 2.0, 3.0, 4.0],
            "eigen_states": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [-sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [0j, 0j, -sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "matrix": [
                [1.5 + 0j, -0.5 + 0j, 0j, 0j],
                [-0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, 3.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, 3.5 + 0j],
            ],
            "matrix_dim": 4,
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
                PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            ],
            "expected_eigen_values": [2.0],
            "expected_projections": [Observable([[1.0 + 0j, 0j], [0j, 1.0 + 0j]])],
        },
        # 2粒子系、縮退あり
        {
            "eigen_values": [2.0, 2.0, -1.0, 1.0],
            "eigen_states": [
                PureQubits([1.0 + 0j, 0j, 0j, 0j]),
                PureQubits([0j, 1.0 + 0j, 0j, 0j]),
                PureQubits([0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
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
            "eigen_values": [1.0, -1.0],
            "eigen_states": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "matrix": [
                [0.5 + 0j, 0.5 + 0j, 0j, 0j],
                [0.5 + 0j, 0.5 + 0j, 0j, 0j],
                [0j, 0j, -0.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, -0.5 + 0j],
            ],
            "matrix_dim": 4,
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
            "eigen_values": [2.0, -2.0, 1.0, -1.0],
            "eigen_states": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, -sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "matrix": [
                [1.5 + 0j, 0.5 + 0j, 0j, 0j],
                [0.5 + 0j, 1.5 + 0j, 0j, 0j],
                [0j, 0j, -1.5 + 0j, -0.5 + 0j],
                [0j, 0j, -0.5 + 0j, -1.5 + 0j],
            ],
            "matrix_dim": 4,
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
            "eigen_values": [1.0, -1.0],
            "eigen_states": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j, 0j, 0j],
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
            "matrix_dim": 8,
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


# @pytest.fixture()
# def observed_basis(orthogonal_qubits):
#     """単一Qubit系に対する観測基底のfixture"""
#     return ObservedBasis(orthogonal_qubits)


# @pytest.fixture(
#     params=[[100.0, -100.0], [100.0, 50.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
# )
# def valid_observed_values(request):
#     """単一Qubit系に対する妥当な観測値のfixture"""
#     return request.param


# @pytest.fixture(params=[[], [100.0, -100.0, 1.0]])
# def invalid_observed_values(request):
#     """単一Qubit系に対する不正な観測値のfixture"""
#     return request.param


# @pytest.fixture(
#     params=[
#         [
#             PureQubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#             PureQubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#             PureQubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#             PureQubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
#             PureQubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
#             PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
#             PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
#             PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
#         ],
#         [
#             PureQubits(
#                 [[[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
#             ),
#             PureQubits(
#                 [[[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
#             ),
#             PureQubits(
#                 [[[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
#             ),
#             PureQubits(
#                 [[[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
#             ),
#             PureQubits(
#                 [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]]]
#             ),
#             PureQubits(
#                 [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]]]
#             ),
#             PureQubits(
#                 [[[0j, 0j], [0j, 0j]], [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]]]
#             ),
#             PureQubits(
#                 [[[0j, 0j], [0j, 0j]], [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]]
#             ),
#         ],
#     ]
# )
# def multi_particles_observed_basis(request):
#     """3粒子Qubit系に対する観測基底のfixture"""
#     return ObservedBasis(request.param)


# @pytest.fixture(
#     params=[
#         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )
# def valid_multi_particles_observed_values(request):
#     """3粒子Qubit系に対する妥当な観測値のfixture"""
#     return request.param


# @pytest.fixture(
#     params=[[], [100.0, -100.0, 10.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
# )
# def invalid_multi_particles_observed_values(request):
#     """3粒子Qubit系に対する不正な観測値のfixture"""
#     return request.param


# @pytest.fixture()
# def observable(valid_observed_value, observe_basis):
#     """単一Qubitに対する観測量のfixture"""
#     return Observable(observe_basis, observe_basis)


# @pytest.fixture(
#     params=[
#         {
#             "observable": Observable(
#                 [100.0, -100.0],
#                 ObservedBasis([PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])]),
#             ),
#             "qubit": PureQubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
#             "expected_value": 40.0,
#         },
#         {
#             "observable": Observable(
#                 [100.0, 50.0],
#                 ObservedBasis([PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])]),
#             ),
#             "qubit": PureQubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
#             "expected_value": 85.0,
#         },
#         {
#             "observable": Observable(
#                 [1.0, 0.0],
#                 ObservedBasis(
#                     [
#                         PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
#                         PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
#                     ]
#                 ),
#             ),
#             "qubit": PureQubits([0 + 0j, 1 + 0j]),
#             "expected_value": 0.5,
#         },
#         {
#             "observable": Observable(
#                 [2.0, 1.0],
#                 ObservedBasis(
#                     [
#                         PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
#                         PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
#                     ]
#                 ),
#             ),
#             "qubit": PureQubits([0 + 0j, 1 + 0j]),
#             "expected_value": 1.5,
#         },
#     ]
# )
# def dict_for_test_expected_value(request):
#     """単一Qubitに対する観測量、観測対象Qubit、期待値の組のfixture"""
#     return request.param


# @pytest.fixture(
#     params=[
#         {
#             "observable": Observable(
#                 [100.0, -100.0],
#                 ObservedBasis(
#                     [
#                         PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
#                         PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
#                     ]
#                 ),
#             ),
#             "qubit": PureQubits([1 + 0j, 0j]),
#             "randomize_seed": 1,
#         },
#         {
#             "observable": Observable(
#                 [10.0, 0.0],
#                 ObservedBasis([PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])]),
#             ),
#             "qubit": PureQubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
#             "randomize_seed": 1,
#         },
#     ]
# )
# def dict_for_test_observation(request):
#     """単一Qubitに対する観測量、観測対象Qubitのfixture"""
#     return request.param


# @pytest.fixture()
# def compound_observable(
#     valid_multi_particles_observed_values, multi_particles_observe_basis
# ):
#     """3粒子Qubit系に対する観測量のfixture"""
#     return Observable(
#         valid_multi_particles_observed_values, multi_particles_observe_basis
#     )


# @pytest.fixture(
#     params=[
#         {
#             "observable": Observable(
#                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
#                 ObservedBasis(
#                     [
#                         PureQubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
#                     ]
#                 ),
#             ),
#             "qubits": PureQubits(
#                 [
#                     [[0j, sqrt(0.25) + 0j], [0j, 0j]],
#                     [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
#                 ]
#             ),
#             "expected_value": 5,
#         }
#     ]
# )
# def dict_for_test_expected_value_with_compound_observable(request):
#     """3粒子Qubit系に対する観測量、観測対象PureQubits、期待値の組のfixture"""
#     return request.param


# @pytest.fixture(
#     params=[
#         {
#             "observable": Observable(
#                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
#                 ObservedBasis(
#                     [
#                         PureQubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
#                         PureQubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
#                     ]
#                 ),
#             ),
#             "qubits": PureQubits(
#                 [
#                     [[sqrt(0.25) + 0j, 0j], [0j, 0j]],
#                     [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
#                 ]
#             ),
#             "randomize_seed": 1,
#         }
#     ]
# )
# def dict_for_test_observation_with_compound_observable(request):
#     """3粒子Qubit系Qubitに対する観測量、観測対象PureQubits、ランダムシードのfixture"""
#     return request.param


# @pytest.fixture(
#     params=[
#         {
#             "observable_group": [
#                 Observable(
#                     [1, 2, 3, 4],
#                     ObservedBasis(
#                         [
#                             PureQubits([[1 + 0j, 0j], [0j, 0j]]),
#                             PureQubits([[0j, 1 + 0j], [0j, 0j]]),
#                             PureQubits([[0j, 0j], [1 + 0j, 0j]]),
#                             PureQubits([[0j, 0j], [0j, 1 + 0j]]),
#                         ]
#                     ),
#                 ),
#                 Observable(
#                     [100, -100, 1000, -1000],
#                     ObservedBasis(
#                         [
#                             PureQubits([[1 + 0j, 0j], [0j, 0j]]),
#                             PureQubits([[0j, 1 + 0j], [0j, 0j]]),
#                             PureQubits([[0j, 0j], [1 + 0j, 0j]]),
#                             PureQubits([[0j, 0j], [0j, 1 + 0j]]),
#                         ]
#                     ),
#                 ),
#             ],
#             "expected_matrix": [
#                 [100 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [0j, -100 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [0j, 0j, 1000 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [
#                     0j,
#                     0j,
#                     0j,
#                     -1000 + 0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                 ],
#                 [0j, 0j, 0j, 0j, 200 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [0j, 0j, 0j, 0j, 0j, -200 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 2000 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     -2000 + 0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                 ],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 300 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, -300 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 3000 + 0j, 0j, 0j, 0j, 0j, 0j],
#                 [
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     -3000 + 0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                 ],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 400 + 0j, 0j, 0j, 0j],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, -400 + 0j, 0j, 0j],
#                 [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 4000 + 0j, 0j],
#                 [
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     0j,
#                     -4000 + 0j,
#                 ],
#             ],
#         }
#     ]
# )
# def dict_for_test_combine_observables(request):
#     """2粒子Qubit系に対する観測量の組および結合後の観測量のfixture"""
#     return request.param
