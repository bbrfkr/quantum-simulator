from math import sqrt

import pytest

from quantum_simulator.base.observable import OrthogonalSystem
from quantum_simulator.base.pure_qubits import PureQubits
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.base.transformer import UnitaryTransformer


@pytest.fixture(
    params=[
        {
            "target": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "matrix": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "matrix_dim": 2,
        },
        {
            "target": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "matrix": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [
                    [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
                    [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]],
                ],
                [
                    [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
                    [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]],
                ],
            ],
            "matrix_dim": 4,
        },
    ]
)
def dict_for_test_unitary_constructor(request):
    """__init__メソッドの正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -1 + 0j]],
        [[1.0 + 0j, 0j, 0j], [0j, 1.0 + 0j, 0j], [0j, 0j, 1.0 + 0j]],
        [
            [1.0 + 0j, 0j, 0j, 0j],
            [0j, -2.0 + 0j, 0j, 0j],
            [0j, 0j, 3.0 + 0j, 0j],
            [0j, 0j, 0j, -4.0 + 0j],
        ],
    ]
)
def dict_for_test_invalid_unitary(request):
    """__init__メソッドの異常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "unitary": UnitaryTransformer([[1.0 + 0j, 0j], [0j, 1.0 + 0j]]),
            "target": Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
            "expected_qubits": Qubits([[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]),
        },
        {
            "unitary": UnitaryTransformer([[0j, 1.0 + 0j], [1.0 + 0j, 0j]]),
            "target": Qubits([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]),
            "expected_qubits": Qubits([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]),
        },
        {
            "unitary": UnitaryTransformer(
                [
                    [1.0 + 0j, 0j, 0j, 0j],
                    [0j, 1.0 + 0j, 0j, 0j],
                    [0j, 0j, 0j, 1.0 + 0j],
                    [0j, 0j, 1.0 + 0j, 0j],
                ]
            ),
            "target": Qubits(
                [
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 1.0 + 0j, 0j],
                    [0j, 0j, 0j, 0j],
                ]
            ),
            "expected_qubits": Qubits(
                [
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 1.0 + 0j],
                ]
            ),
        },
    ]
)
def test_for_success_operate(request):
    """operateメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "onb_0": OrthogonalSystem(
                [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
            ),
            "onb_1": OrthogonalSystem(
                [
                    PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                    PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                ]
            ),
            "matrix": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "matrix_dim": 2,
        },
        {
            "onb_0": OrthogonalSystem(
                [PureQubits([0j, 1.0 + 0j]), PureQubits([1.0 + 0j, 0j])]
            ),
            "onb_1": OrthogonalSystem(
                [
                    PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                    PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                ]
            ),
            "matrix": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "matrix_dim": 2,
        },
    ]
)
def dict_for_test_create_from_onb(request):
    """create_from_onbメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "unitary_0": UnitaryTransformer([[1.0 + 0j, 0j], [0j, 1.0 + 0j]]),
            "unitary_1": UnitaryTransformer(
                [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]
            ),
            "matrix": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [
                    [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
                    [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]],
                ],
                [
                    [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
                    [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]],
                ],
            ],
            "matrix_dim": 4,
        },
    ]
)
def dict_for_test_unitary_combine(request):
    """combineメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "unitary_list": [
                UnitaryTransformer([[1.0 + 0j, 0j], [0j, 1.0 + 0j]]),
                UnitaryTransformer([[0j, 1.0 + 0j], [1.0 + 0j, 0j]]),
                UnitaryTransformer(
                    [
                        [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                        [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                    ]
                ),
            ],
            "matrix": [
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j, 0j, 0j],
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                [0j, 0j, 0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j, 0j],
            ],
            "ndarray": [
                [
                    [
                        [
                            [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
                            [[0j, 0j], [0j, 0j]],
                        ],
                        [
                            [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]],
                            [[0j, 0j], [0j, 0j]],
                        ],
                    ],
                    [
                        [
                            [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
                            [[0j, 0j], [0j, 0j]],
                        ],
                        [
                            [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]],
                            [[0j, 0j], [0j, 0j]],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [[0j, 0j], [0j, 0j]],
                            [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
                        ],
                        [
                            [[0j, 0j], [0j, 0j]],
                            [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]],
                        ],
                    ],
                    [
                        [
                            [[0j, 0j], [0j, 0j]],
                            [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
                        ],
                        [
                            [[0j, 0j], [0j, 0j]],
                            [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]],
                        ],
                    ],
                ],
            ],
            "matrix_dim": 8,
        },
    ]
)
def dict_for_test_unitary_multiple_combine(request):
    """multiple_combineメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "unitary_0": UnitaryTransformer([[0j, 1.0 + 0j], [1.0 + 0j, 0j]]),
            "unitary_1": UnitaryTransformer(
                [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]
            ),
            "matrix": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            ],
            "matrix_dim": 2,
        },
    ]
)
def dict_for_test_unitary_compose(request):
    """combineメソッドに対する正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "unitary_list": [
                UnitaryTransformer([[1.0 + 0j, 0j], [0j, -1.0 + 0j]]),
                UnitaryTransformer([[0j, 1.0 + 0j], [1.0 + 0j, 0j]]),
                UnitaryTransformer(
                    [
                        [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                        [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                    ]
                ),
            ],
            "matrix": [
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "ndarray": [
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
                [-sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "matrix_dim": 2,
        },
    ]
)
def dict_for_test_unitary_multiple_compose(request):
    """combineメソッドに対する正常系テスト用fixture"""
    return request.param
