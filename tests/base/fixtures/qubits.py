from math import sqrt

import pytest

from quantum_simulator.base.pure_qubits import PureQubits
from quantum_simulator.base.qubits import Qubits, combine


@pytest.fixture(
    params=[
        # 行列形式
        [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
        [
            [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            [0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j],
            [0.5 + 0j, 0j, 0j, 0.5 + 0j],
        ],
        [
            [0.25 + 0j, 0j, 0.25 + 0j, 0j, 0j, 0.25 + 0j, 0j, 0.25 + 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0.25 + 0j, 0j, 0.25 + 0j, 0j, 0j, 0.25 + 0j, 0j, 0.25 + 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0.25 + 0j, 0j, 0.25 + 0j, 0j, 0j, 0.25 + 0j, 0j, 0.25 + 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0.25 + 0j, 0j, 0.25 + 0j, 0j, 0j, 0.25 + 0j, 0j, 0.25 + 0j],
        ],
        # ndarray形式
        [
            [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
            [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
        ],
        [
            [
                [
                    [
                        [[0.25 + 0j, 0j], [0.25 + 0j, 0j]],
                        [[0j, 0.25 + 0j], [0j, 0.25 + 0j]],
                    ],
                    [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                ],
                [
                    [
                        [[0.25 + 0j, 0j], [0.25 + 0j, 0j]],
                        [[0j, 0.25 + 0j], [0j, 0.25 + 0j]],
                    ],
                    [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                ],
            ],
            [
                [
                    [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    [
                        [[0.25 + 0j, 0j], [0.25 + 0j, 0j]],
                        [[0j, 0.25 + 0j], [0j, 0.25 + 0j]],
                    ],
                ],
                [
                    [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    [
                        [[0.25 + 0j, 0j], [0.25 + 0j, 0j]],
                        [[0j, 0.25 + 0j], [0j, 0.25 + 0j]],
                    ],
                ],
            ],
        ],
    ]
)
def valid_qubits_array(request):
    """妥当なQubitsに対するarrayのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # ベクトル
        [1.0 + 0j, 0j],
        # 正方行列でない行列
        [
            [sqrt(1 / 8) + 0j, 0j],
            [0j, sqrt(1 / 8) + 0j],
            [sqrt(1 / 8) + 0j, 0j],
            [0j, sqrt(1 / 8) + 0j],
            [sqrt(1 / 8) + 0j, 0j],
            [0j, sqrt(1 / 8) + 0j],
            [sqrt(1 / 8) + 0j, 0j],
            [0j, sqrt(1 / 8) + 0j],
        ],
        # 正方行列だが、次元が2^nでない行列
        [
            [sqrt(1 / 3) + 0j, 0j, 0j],
            [0j, sqrt(1 / 3) + 0j, 0j],
            [0j, 0j, sqrt(1 / 3) + 0j],
        ],
        # shapeの要素数が2の倍数でないndarray
        [
            [[sqrt(1 / 4) + 0j, 0j, 0j, 0j], [0j, 0j, 0j, sqrt(1 / 4) + 0j]],
            [[sqrt(1 / 4) + 0j, 0j, 0j, 0j], 0j, 0j, 0j, sqrt(1 / 4) + 0j],
        ],
        # shapeの各要素が2でないndarray
        [
            [[sqrt(1 / 4) + 0j, 0j, 0j, 0j], [0j, 0j, 0j, sqrt(1 / 4) + 0j]],
            [[sqrt(1 / 4) + 0j, 0j, 0j, 0j], [0j, 0j, 0j, sqrt(1 / 4) + 0j]],
        ],
    ]
)
def invalid_qubits_array(request):
    """妥当でないQubitsに対するarrayのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "target": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "matrix": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "ndarray": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
        },
        # 2粒子行列形式
        {
            "target": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "ndarray": [
                [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
            ],
        },
        # 2粒子ndarray形式
        {
            "target": [
                [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
            ],
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "ndarray": [
                [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
            ],
        },
    ]
)
def dict_for_test_qubits_resolve_arrays(request):
    """resolve_arraysメソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "target": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "eigen_values": [0.3 + 0j, 0.7 + 0j],
            "eigen_states": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
        },
        {
            "target": [[0.75 + 0j, 0.25 + 0j], [0.25 + 0j, 0.25 + 0j]],
            "eigen_values": [(2 + sqrt(2)) / 4 + 0j, (2 - sqrt(2)) / 4 + 0j],
            "eigen_states": [
                [
                    (1 + sqrt(2)) * (1 / sqrt(4 + 2 * sqrt(2))) + 0j,
                    (1 / sqrt(4 + 2 * sqrt(2))) + 0j,
                ],
                [
                    (1 - sqrt(2)) * (1 / sqrt(4 - 2 * sqrt(2))) + 0j,
                    (1 / sqrt(4 - 2 * sqrt(2))) + 0j,
                ],
            ],
        },
        # 2粒子Qubit系
        {
            "target": [
                [0.5 + 0j, 0j, 0.25 + 0j, 0.25 + 0j],
                [0j, 0j, 0j, 0j],
                [0.25 + 0j, 0j, 0.25 + 0j, 0j],
                [0.25 + 0j, 0j, 0j, 0.25 + 0j],
            ],
            "eigen_values": [0.25, 0.75],
            "eigen_states": [
                [0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                [sqrt(2 / 3) + 0j, 0j, sqrt(1 / 6) + 0j, sqrt(1 / 6) + 0j],
            ],
        },
    ]
)
def dict_for_test_resolve_eigen(request):
    """resolve_eigenメソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "target": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "eigen_values": [0.3 + 0j, 0.7 + 0j],
            "eigen_states": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "matrix": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "matrix_dim": 2,
            "ndarray": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "qubit_count": 1,
            "is_pure": False,
        },
        {
            "target": [[0.75 + 0j, 0.25 + 0j], [0.25 + 0j, 0.25 + 0j]],
            "eigen_values": [(2 + sqrt(2)) / 4 + 0j, (2 - sqrt(2)) / 4 + 0j],
            "eigen_states": [
                [
                    (1 + sqrt(2)) * (1 / sqrt(4 + 2 * sqrt(2))) + 0j,
                    (1 / sqrt(4 + 2 * sqrt(2))) + 0j,
                ],
                [
                    (1 - sqrt(2)) * (1 / sqrt(4 - 2 * sqrt(2))) + 0j,
                    (1 / sqrt(4 - 2 * sqrt(2))) + 0j,
                ],
            ],
            "matrix": [[0.75 + 0j, 0.25 + 0j], [0.25 + 0j, 0.25 + 0j]],
            "matrix_dim": 2,
            "ndarray": [[0.75 + 0j, 0.25 + 0j], [0.25 + 0j, 0.25 + 0j]],
            "qubit_count": 1,
            "is_pure": False,
        },
        # 2粒子Qubit系
        {
            "target": [
                [0.5 + 0j, 0j, 0.25 + 0j, 0.25 + 0j],
                [0j, 0j, 0j, 0j],
                [0.25 + 0j, 0j, 0.25 + 0j, 0j],
                [0.25 + 0j, 0j, 0j, 0.25 + 0j],
            ],
            "eigen_values": [0.25, 0.75],
            "eigen_states": [
                [0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                [sqrt(2 / 3) + 0j, 0j, sqrt(1 / 6) + 0j, sqrt(1 / 6) + 0j],
            ],
            "matrix": [
                [0.5 + 0j, 0j, 0.25 + 0j, 0.25 + 0j],
                [0j, 0j, 0j, 0j],
                [0.25 + 0j, 0j, 0.25 + 0j, 0j],
                [0.25 + 0j, 0j, 0j, 0.25 + 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0.5 + 0j, 0j], [0.25 + 0j, 0.25 + 0j]], [[0j, 0j], [0j, 0j]]],
                [
                    [[0.25 + 0j, 0j], [0.25 + 0j, 0j]],
                    [[0.25 + 0j, 0j], [0j, 0.25 + 0j]],
                ],
            ],
            "qubit_count": 2,
            "is_pure": False,
        },
        {
            "target": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "eigen_values": [1.0],
            "eigen_states": [[sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]],
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
            ],
            "qubit_count": 2,
            "is_pure": True,
        },
    ]
)
def dict_for_test_qubits_constructor(request):
    """__init__メソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [0.5, 0.3, 0.2],
            "qubits_list": [
                PureQubits([1.0 + 0j, 0j]),
                PureQubits([0j, 1.0 + 0j]),
                Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            ],
            "eigen_values": [0.5, 0.5],
            "eigen_states": [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
            "matrix": [[0.5 + 0j, 0j], [0j, 0.5 + 0j]],
            "matrix_dim": 2,
            "ndarray": [[0.5 + 0j, 0j], [0j, 0.5 + 0j]],
            "qubit_count": 1,
            "is_pure": False,
        },
        {
            "probabilities": [0.5, 0.5],
            "qubits_list": [
                PureQubits([sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]),
                Qubits(
                    [
                        [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                    ]
                ),
            ],
            "eigen_values": [1.0],
            "eigen_states": [[sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]],
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
            ],
            "qubit_count": 2,
            "is_pure": True,
        },
    ]
)
def dict_for_test_create_from_qubits_list(request):
    """create_from_qubits_listメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [0.2, 0.5],
            "qubits_list": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
        },
        {
            "probabilities": [-0.1, 0.4],
            "qubits_list": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
        },
    ]
)
def invalid_probabilities_and_qubits_list(request):
    """
    create_from_qubits_listメソッドテスト用の異常系fixture
    (不正な確率リスト)
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [0.5, 0.3, 0.2],
            "qubits_list": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
        },
        {
            "probabilities": [1.0],
            "qubits_list": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
        },
    ]
)
def not_match_count_probabilities_and_qubits_list(request):
    """
    create_from_qubits_listメソッドテスト用の異常系fixture
    (リスト要素数不一致)
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits_list": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
            "eigen_values": [1.0],
            "eigen_states": [[0j, 1.0 + 0j, 0j, 0j]],
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0j, 0j], [0j, 0j]], [[0j, 1.0 + 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
            ],
            "qubit_count": 2,
            "is_pure": True,
        },
        {
            "qubits_list": [
                PureQubits([1.0 + 0j, 0j]),
                Qubits(
                    [
                        [0.25 + 0j, 0j, 0j, 0.25 + 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0.5 + 0j, 0j],
                        [0.25 + 0j, 0j, 0j, 0.25 + 0j],
                    ]
                ),
            ],
            "eigen_values": [0.5, 0.5],
            "eigen_states": [
                [sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j],
            ],
            "matrix": [
                [0.25 + 0j, 0j, 0j, 0.25 + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0j, 0j, 0j, 0j, 0j],
                [0.25 + 0j, 0j, 0j, 0.25 + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            ],
            "matrix_dim": 8,
            "ndarray": [
                [
                    [
                        [[[0.25 + 0j, 0j], [0j, 0.25 + 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                    [
                        [[[0j, 0j], [0.5 + 0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0.25 + 0j, 0j], [0j, 0.25 + 0j]], [[0j, 0j], [0j, 0j]]],
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
            "qubit_count": 3,
            "is_pure": False,
        },
    ]
)
def dict_for_test_qubits_combine(request):
    """combineメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits": combine(
                combine(PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 1,
            "eigen_values": [0.5, 0.5],
            "eigen_states": [[1.0 + 0j, 0j, 0j, 0j], [0j, 1.0 + 0j, 0j, 0j]],
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0j],
                [0j, 0.5 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0.5 + 0j, 0j], [0j, 0j]], [[0j, 0.5 + 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
            ],
            "qubit_count": 2,
            "is_pure": False,
        },
        {
            "qubits": combine(
                combine(PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 0,
            "eigen_values": [0.5, 0.5],
            "eigen_states": [[0j, 0j, 1.0 + 0j, 0j], [0j, 0j, 0j, 1.0 + 0j]],
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0j],
                [0j, 0j, 0j, 0.5 + 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0.5 + 0j, 0j]], [[0j, 0j], [0j, 0.5 + 0j]]],
            ],
            "qubit_count": 2,
            "is_pure": False,
        },
        {
            "qubits": combine(
                combine(PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 2,
            "eigen_values": [1.0],
            "eigen_states": [[0j, 1.0 + 0j, 0j, 0j]],
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
            ],
            "matrix_dim": 4,
            "ndarray": [
                [[[0j, 0j], [0j, 0j]], [[0j, 1.0 + 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
            ],
            "qubit_count": 2,
            "is_pure": True,
        },
    ]
)
def dict_for_test_reduction(request):
    """reductionメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits": combine(
                combine(PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": -1,
        },
        {
            "qubits": combine(
                combine(PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 3,
        },
    ]
)
def invalid_reduction(request):
    """reductionメソッドテスト用の異常系fixture"""
    return request.param
