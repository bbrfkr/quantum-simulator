from math import sqrt

import numpy as np
import pytest

from src.base.qubits import Qubits


# 妥当な単一qubitに対する確率振幅のfixture
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


# 妥当な単一qubitのfixture
@pytest.fixture()
def valid_qubit(valid_qubit_amp):
    amplitudes = np.array(valid_qubit_amp)
    return Qubits(amplitudes)


# 不正な単一qubitに対する確率振幅のfixture
@pytest.fixture(
    params=[
        [0 + 0j, 0j],
        [sqrt(0.3) + 0j, sqrt(0.3) + 0j],
        [sqrt(0.3) * 1j, sqrt(0.3) * 1j],
        [sqrt(0.6) + 0j, sqrt(0.6) + 0j],
        [sqrt(0.6) * 1j, sqrt(0.6) * 1j],
        [1 + 0j],
        [sqrt(0.25) + 0j, sqrt(0.25) + 0j, sqrt(0.25) + 0j, sqrt(0.25) + 0j],
        [],
    ]
)
def invalid_qubit_amp(request):
    return request.param


# 直交する単一qubit同士のfixture
@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [0j, 1 + 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, -sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.6) + 0j, -sqrt(0.4) * 1j]],
    ]
)
def orthogonal_qubits(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 直交しない単一qubit同士のfixture
@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [1 + 0j, 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.4) + 0j, -sqrt(0.6) * 1j]],
    ]
)
def non_orthogonal_qubits(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 妥当なQubit群に対する確率振幅のfixture
# 2粒子: |00>
# 3粒子: |010>
# 4粒子: |1001>
# 重ね合わせ2粒子(EPR pair): sqrt(0.5)|00> + sqrt(0.5)|11>
@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [0j, 0j]],
        [[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]],
        [[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]],
        [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]],
    ]
)
def valid_qubits_amp(request):
    return request.param


# 妥当なQubit群のfixture
@pytest.fixture()
def valid_qubits(valid_qubits_amp):
    amplitudes = np.array(valid_qubits_amp)
    return Qubits(amplitudes)


# 粒子間結合テスト用fixture
# test1: |0> x |+> = sqrt(0.5)|00> + sqrt(0.5)|01>
# test2: |+> x |-> = 0.5|00> - 0.5|01> + 0.5|10> - 0.5|11>
# test2: |0-> x |1> = sqrt(0.5)|001> - sqrt(0.5)|011>
@pytest.fixture(
    params=[
        {
            "qubits_group": [[1 + 0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
            "result": [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
        },
        {
            "qubits_group": [
                [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
                [sqrt(0.5) + 0j, -sqrt(0.5) + 0j],
            ],
            "result": [[0.5 + 0j, -0.5 + 0j], [0.5 + 0j, -0.5 + 0j]],
        },
        {
            "qubits_group": [
                [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]],
                [0j, 1 + 0j],
            ],
            "result": [
                [[0j, sqrt(0.5) + 0j], [0j, -sqrt(0.5) + 0j]],
                [[0j, 0j], [0j, 0j]],
            ],
        },
    ]
)
def dict_test_for_combine(request):
    test_dict = {
        "qubits_group": [
            Qubits(np.array(amplitudes)) for amplitudes in request.param["qubits_group"]
        ],
        "result": request.param["result"],
    }
    return test_dict


# 直交する二つのQubit群同士のfixture
# test1: <1+|0+>
# test2: <0+|0->
@pytest.fixture(
    params=[
        [
            [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
            [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
        ],
        [
            [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]],
            [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],
        ],
    ]
)
def orthogonal_two_qubits_groups(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 直交しない二つのQubit群のfixture
# test1: {|-+>, |1+>} (<1+|-+>)
# test2: {|0->, |01>} (<01|0->)
@pytest.fixture(
    params=[
        [
            [[0.5 + 0j, 0.5 + 0j], [-0.5 + 0j, -0.5 + 0j]],
            [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
        ],
        [[[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]],
    ]
)
def non_orthogonal_two_qubits_groups(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 異なるQubit数の二つのQubit群のfixture
# test1: <1|0+>
# test2: <0+|0>
@pytest.fixture(
    params=[
        [
            [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
            [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]],
        ],
        [[[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]], [1j, 0j]],
    ]
)
def not_match_counts_two_qubits_groups(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 互いに直交する二つより多いQubit群同士のfixture
# test1: {|00>, |01>, |10>, |11>}
@pytest.fixture(
    params=[
        [
            [[1 + 0j, 0j], [0j, 0j]],
            [[0j, 1 + 0j], [0j, 0j]],
            [[0j, 0j], [1 + 0j, 0j]],
            [[0j, 0j], [0j, 1 + 0j]],
        ]
    ]
)
def orthogonal_multiple_qubits_groups(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 互いに直交しない二つより多いQubit群同士のfixture
# test1: {|00>, |01>, |10>, |11>, |00>}
@pytest.fixture(
    params=[
        [
            [[1 + 0j, 0j], [0j, 0j]],
            [[0j, 1 + 0j], [0j, 0j]],
            [[0j, 0j], [1 + 0j, 0j]],
            [[0j, 0j], [0j, 1 + 0j]],
            [[1 + 0j, 0j], [0j, 0j]],
        ]
    ]
)
def non_orthogonal_multiple_qubits_groups(request):
    amplitudes_list = [np.array(amplitudes) for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# # 観測基底のfixture
# @pytest.fixture()
# def observe_basis(orthogonal_qubits):
#     return ObserveBasis(orthogonal_qubits)


# # 妥当な観測値のfixture
# @pytest.fixture(params=[[100.0, -100.0], [100.0, 50.0], [1.0, 0.0], [0.0, 1.0]])
# def valid_observed_value(request):
#     return request.param


# # 不正な観測値のfixture
# @pytest.fixture(params=[[0.0, 0.0], [100.0, 100.0], [1.0, 1.0], [-50.0, -50.0]])
# def invalid_observed_value(request):
#     return request.param


# # 観測量のfixture
# @pytest.fixture()
# def observable(valid_observed_value, observe_basis):
#     return Observable(valid_observed_value[0], valid_observed_value[1], observe_basis)


# # 期待値をテストするための観測量、観測対象Qubit、期待値の組
# @pytest.fixture(
#     params=[
#         {
#             "observable": Observable(
#                 100.0, -100.0, ObserveBasis(Qubit(1 + 0j, 0j), Qubit(0j, 1 + 0j))
#             ),
#             "qubit": Qubit(sqrt(0.7) + 0j, sqrt(0.3) + 0j),
#             "expected_value": 40.0,
#         },
#         {
#             "observable": Observable(
#                 100.0, 50.0, ObserveBasis(Qubit(1 + 0j, 0j), Qubit(0j, 1 + 0j))
#             ),
#             "qubit": Qubit(sqrt(0.7) + 0j, sqrt(0.3) + 0j),
#             "expected_value": 85.0,
#         },
#         {
#             "observable": Observable(
#                 1.0,
#                 0.0,
#                 ObserveBasis(
#                     Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j),
#                     Qubit(sqrt(0.5) + 0j, -sqrt(0.5) + 0j),
#                 ),
#             ),
#             "qubit": Qubit(0 + 0j, 1 + 0j),
#             "expected_value": 0.5,
#         },
#         {
#             "observable": Observable(
#                 2.0,
#                 1.0,
#                 ObserveBasis(
#                     Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j),
#                     Qubit(sqrt(0.5) + 0j, -sqrt(0.5) + 0j),
#                 ),
#             ),
#             "qubit": Qubit(0 + 0j, 1 + 0j),
#             "expected_value": 1.5,
#         },
#     ]
# )
# def dict_for_test_expected_value(request):
#     return request.param


# # 観測結果をテストするための観測量、観測対象Qubit
# @pytest.fixture(
#     params=[
#         {
#             "observable": Observable(
#                 100.0,
#                 -100.0,
#                 ObserveBasis(
#                     Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j),
#                     Qubit(sqrt(0.5) + 0j, -sqrt(0.5) + 0j),
#                 ),
#             ),
#             "qubit": Qubit(1 + 0j, 0j),
#         }
#     ]
# )
# def dict_for_test_observation(request):
#     return request.param
