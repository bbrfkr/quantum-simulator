from math import sqrt

import pytest

from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits
from quantum_simulator.base.qubits import Qubits, combine, generalize, multiple_combine


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
            "matrix": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "qubit_count": 1,
        },
        {
            "target": [[0.75 + 0j, 0.25 + 0j], [0.25 + 0j, 0.25 + 0j]],
            "matrix": [[0.75 + 0j, 0.25 + 0j], [0.25 + 0j, 0.25 + 0j]],
            "qubit_count": 1,
        },
        # 2粒子Qubit系
        {
            "target": [
                [0.5 + 0j, 0j, 0.25 + 0j, 0.25 + 0j],
                [0j, 0j, 0j, 0j],
                [0.25 + 0j, 0j, 0.25 + 0j, 0j],
                [0.25 + 0j, 0j, 0j, 0.25 + 0j],
            ],
            "matrix": [
                [0.5 + 0j, 0j, 0.25 + 0j, 0.25 + 0j],
                [0j, 0j, 0j, 0j],
                [0.25 + 0j, 0j, 0.25 + 0j, 0j],
                [0.25 + 0j, 0j, 0j, 0.25 + 0j],
            ],
            "qubit_count": 2,
        },
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
            "qubit_count": 2,
        },
    ]
)
def dict_for_test_qubits_constructor(request):
    """__init__メソッドテスト用のfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "target": PureQubits([1.0 + 0j, 0j]),
            "matrix": [[1.0 + 0j, 0j], [0j, 0j]],
            "qubit_count": 1,
        },
        {
            "target": PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            "matrix": [[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]],
            "qubit_count": 1,
        },
        # 2粒子系
        {
            "target": PureQubits([sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]),
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "qubit_count": 2,
        },
    ]
)
def dict_for_test_generalize(request):
    """generalizeメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "target": Qubits([[1.0 + 0j, 0j], [0j, 0j]]),
            "vector": [1.0 + 0j, 0j],
            "qubit_count": 1,
        },
        {
            "target": Qubits([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]),
            "vector": [-sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            "qubit_count": 1,
        },
        # 2粒子系
        {
            "target": Qubits(
                [
                    [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                    [0j, 0j, 0j, 0j],
                    [0j, 0j, 0j, 0j],
                    [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                ]
            ),
            "vector": [-sqrt(0.5) + 0j, 0j, 0j, -sqrt(0.5) + 0j],
            "qubit_count": 2,
        },
    ]
)
def dict_for_test_specialize(request):
    """specializeメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [0.5, 0.3, 0.2],
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
                Qubits([[0j, 0j], [0j, 1.0 + 0j]]),
            ],
            "matrix": [[0.5 + 0j, 0j], [0j, 0.5 + 0j]],
            "qubit_count": 1,
        },
        {
            "probabilities": [0.5, 0.5],
            "qubits_list": [
                generalize(PureQubits([sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j])),
                Qubits(
                    [
                        [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0j, 0j],
                        [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                    ]
                ),
            ],
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0.5 + 0j, 0j, 0j, 0.5 + 0j],
            ],
            "qubit_count": 2,
        },
    ]
)
def dict_for_test_convex_combination(request):
    """convex_combinationメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [0.2, 0.5],
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
            ],
        },
        {
            "probabilities": [-0.1, 0.4],
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
            ],
        },
    ]
)
def invalid_probabilities_and_qubits_list(request):
    """
    convex_combinationメソッドテスト用の異常系fixture
    (不正な確率リスト)
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "probabilities": [0.5, 0.3, 0.2],
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
            ],
        },
        {
            "probabilities": [1.0],
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
            ],
        },
    ]
)
def not_match_count_probabilities_and_qubits_list(request):
    """
    convex_combinationメソッドテスト用の異常系fixture
    (リスト要素数不一致)
    """
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit
        {
            "probabilities": [0.3, 0.7],
            "ons": OrthogonalSystem(
                [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
            ),
            "matrix": [[0.3 + 0j, 0j], [0j, 0.7 + 0j]],
            "qubit_count": 1,
        },
        {
            "probabilities": [0.25, 0.75],
            "ons": OrthogonalSystem(
                [
                    PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                    PureQubits([-sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                ]
            ),
            "matrix": [[0.5 + 0j, -0.25 + 0j], [-0.25 + 0j, 0.5 + 0j]],
            "qubit_count": 1,
        },
    ]
)
def dict_for_test_create_from_ons(request):
    """create_for_onsメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
            ],
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
            ],
            "qubit_count": 2,
        },
        {
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                Qubits(
                    [
                        [0.25 + 0j, 0j, 0j, 0.25 + 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0.5 + 0j, 0j],
                        [0.25 + 0j, 0j, 0j, 0.25 + 0j],
                    ]
                ),
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
            "qubit_count": 3,
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
                combine(
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                ),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 1,
            "matrix": [
                [0.5 + 0j, 0j, 0j, 0j],
                [0j, 0.5 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
            ],
            "qubit_count": 2,
        },
        {
            "qubits": combine(
                combine(
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                ),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 0,
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0j],
                [0j, 0j, 0j, 0.5 + 0j],
            ],
            "qubit_count": 2,
        },
        {
            "qubits": combine(
                combine(
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                ),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 2,
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 1.0 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
            ],
            "qubit_count": 2,
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
                combine(
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                ),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": -1,
        },
        {
            "qubits": combine(
                combine(
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                ),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ),
            "target_particle": 3,
        },
    ]
)
def invalid_reduction(request):
    """reductionメソッドテスト用の異常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                generalize(PureQubits([0j, 1.0 + 0j])),
                Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
            ],
            "matrix": [
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0.5 + 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            ],
            "qubit_count": 3,
        },
        {
            "qubits_list": [
                generalize(PureQubits([1.0 + 0j, 0j])),
                Qubits(
                    [
                        [0.25 + 0j, 0j, 0j, 0.25 + 0j],
                        [0j, 0j, 0j, 0j],
                        [0j, 0j, 0.5 + 0j, 0j],
                        [0.25 + 0j, 0j, 0j, 0.25 + 0j],
                    ]
                ),
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
            "qubit_count": 3,
        },
    ]
)
def dict_for_test_qubits_multiple_combine(request):
    """combineメソッドテスト用の正常系fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits": multiple_combine(
                [
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                    Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
                ]
            ),
            "target_particles": [0, 1],
            "matrix": [[0.5 + 0j, 0j], [0j, 0.5 + 0j]],
            "qubit_count": 1,
        },
        {
            "qubits": multiple_combine(
                [
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                    Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
                ]
            ),
            "target_particles": [0],
            "matrix": [
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j],
                [0j, 0j, 0.5 + 0j, 0j],
                [0j, 0j, 0j, 0.5 + 0j],
            ],
            "qubit_count": 2,
        },
        {
            "qubits": multiple_combine(
                [
                    generalize(PureQubits([1.0 + 0j, 0j])),
                    generalize(PureQubits([0j, 1.0 + 0j])),
                    Qubits([[0.5 + 0j, 0j], [0j, 0.5 + 0j]]),
                ]
            ),
            "target_particles": [0, 2],
            "matrix": [[0j, 0j], [0j, 1.0 + 0j]],
            "qubit_count": 1,
        },
    ]
)
def dict_for_test_multiple_reduction(request):
    """reductionメソッドテスト用の正常系fixture"""
    return request.param
