from math import sqrt

import numpy as np
import pytest

from src.base.observable import Observable, ObservedBasis
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
    amplitudes = valid_qubit_amp
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
    amplitudes_list = [amplitudes for amplitudes in request.param]
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
    amplitudes_list = [amplitudes for amplitudes in request.param]
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
    amplitudes = valid_qubits_amp
    return Qubits(amplitudes)


# 単一Qubitに対する射影作用素のfixture
@pytest.fixture(
    params=[
        {
            "qubit": Qubits([1.0 + 0j, 0.0 + 0j]),
            "projection": [[1.0 + 0j, 0j], [0j, 0j]],
        },
        {
            "qubit": Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
            "projection": [[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]],
        },
    ]
)
def proj_for_valid_qubit(request):
    return request.param


# 複数Qubitに対する射影作用素のfixture
# 2粒子: |00>
# 3粒子: |010>
# 4粒子: |1001>
# 重ね合わせ2粒子(EPR pair): sqrt(0.5)|00> + sqrt(0.5)|11>
@pytest.fixture(
    params=[
        {
            "qubits": Qubits([[1 + 0j, 0j], [0j, 0j]]),
            "projection": [
                [[[1 + 0j, 0j], [0j, 0j]], [[0 + 0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0 + 0j, 0j], [0j, 0j]]],
            ],
        },
        {
            "qubits": Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            "projection": [
                [
                    [
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                    [
                        [[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
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
        {
            "qubits": Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
            "projection": [
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
                [
                    [
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]],
                    ],
                    [
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                        [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
                    ],
                ],
            ],
        },
        {
            "qubits": Qubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
            "projection": [
                [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
                [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
            ],
        },
    ]
)
def proj_for_valid_qubits(request):
    return request.param


# 粒子間結合テスト用fixture
# test1: |0> x |+> = sqrt(0.5)|00> + sqrt(0.5)|01>
# test2: |+> x |-> = 0.5|00> - 0.5|01> + 0.5|10> - 0.5|11>
# test3: |0-> x |1> = sqrt(0.5)|001> - sqrt(0.5)|011>
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
            Qubits(amplitudes) for amplitudes in request.param["qubits_group"]
        ],
        "result": request.param["result"],
    }
    return test_dict


# 直交する二つのQubit群同士のfixture
# test1: |0+>, |1+>
# test2: |0->, |0+>
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
    amplitudes_list = [amplitudes for amplitudes in request.param]
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
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 異なるQubit数の二つのQubit群のfixture
# test1: |0+>, |1>
# test2: |0>, |0+>
@pytest.fixture(
    params=[
        [[[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]], [0j, 1.0 + 0j],],
        [[1.0 + 0j, 0j], [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]],],
    ]
)
def not_match_counts_two_qubits_groups(request):
    amplitudes_list = [amplitudes for amplitudes in request.param]
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
    amplitudes_list = [amplitudes for amplitudes in request.param]
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
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


# 単一Qubit系に対する観測基底のfixture
@pytest.fixture()
def observed_basis(orthogonal_qubits):
    return ObservedBasis(orthogonal_qubits)


# 単一Qubit系に対する妥当な観測値のfixture
@pytest.fixture(
    params=[[100.0, -100.0], [100.0, 50.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
)
def valid_observed_values(request):
    return request.param


# 単一Qubit系に対する不正な観測値のfixture
@pytest.fixture(params=[[], [100.0, -100.0, 1.0]])
def invalid_observed_values(request):
    return request.param


# 3粒子Qubit系に対する観測基底のfixture
@pytest.fixture(
    params=[
        [
            Qubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
            Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
        ],
        [
            Qubits(
                [[[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]], [[0j, 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[sqrt(0.5) + 0j, -sqrt(0.5) + 0j], [0j, 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [sqrt(0.5) + 0j, sqrt(0.5) + 0j]]]
            ),
            Qubits(
                [[[0j, 0j], [0j, 0j]], [[0j, 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]]
            ),
        ],
    ]
)
def multi_particles_observed_basis(request):
    return ObservedBasis(request.param)


# 3粒子Qubit系に対する妥当な観測値のfixture
@pytest.fixture(
    params=[
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
def valid_multi_particles_observed_values(request):
    return request.param


# 3粒子Qubit系に対する不正な観測値のfixture
@pytest.fixture(
    params=[[], [100.0, -100.0, 10.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
)
def invalid_multi_particles_observed_values(request):
    return request.param


# 単一Qubitに対する観測量のfixture
@pytest.fixture()
def observable(valid_observed_value, observe_basis):
    return Observable(observe_basis, observe_basis)


# 単一Qubitに対する観測量、観測対象Qubit、期待値の組のfixture
@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [100.0, -100.0],
                ObservedBasis([Qubits([1 + 0j, 0j]), Qubits([0j, 1 + 0j])]),
            ),
            "qubit": Qubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
            "expected_value": 40.0,
        },
        {
            "observable": Observable(
                [100.0, 50.0],
                ObservedBasis([Qubits([1 + 0j, 0j]), Qubits([0j, 1 + 0j])]),
            ),
            "qubit": Qubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
            "expected_value": 85.0,
        },
        {
            "observable": Observable(
                [1.0, 0.0],
                ObservedBasis(
                    [
                        Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        Qubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "qubit": Qubits([0 + 0j, 1 + 0j]),
            "expected_value": 0.5,
        },
        {
            "observable": Observable(
                [2.0, 1.0],
                ObservedBasis(
                    [
                        Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        Qubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "qubit": Qubits([0 + 0j, 1 + 0j]),
            "expected_value": 1.5,
        },
    ]
)
def dict_for_test_expected_value(request):
    return request.param


# 単一Qubitに対する観測量、観測対象Qubitのfixture
@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [100.0, -100.0],
                ObservedBasis(
                    [
                        Qubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                        Qubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
                    ]
                ),
            ),
            "qubit": Qubits([1 + 0j, 0j]),
            "randomize_seed": 1,
        },
        {
            "observable": Observable(
                [10.0, 0.0],
                ObservedBasis([Qubits([1 + 0j, 0j]), Qubits([0j, 1 + 0j]),]),
            ),
            "qubit": Qubits([sqrt(0.7) + 0j, sqrt(0.3) + 0j]),
            "randomize_seed": 1,
        },
    ]
)
def dict_for_test_observation(request):
    return request.param


# 3粒子Qubit系に対する観測量のfixture
@pytest.fixture()
def compound_observable(
    valid_multi_particles_observed_values, multi_particles_observe_basis
):
    return Observable(
        valid_multi_particles_observed_values, multi_particles_observe_basis
    )


# 3粒子Qubit系に対する観測量、観測対象Qubits、期待値の組のfixture
@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ObservedBasis(
                    [
                        Qubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
                    ]
                ),
            ),
            "qubits": Qubits(
                [
                    [[0j, sqrt(0.25) + 0j], [0j, 0j]],
                    [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
                ]
            ),
            "expected_value": 5,
        },
    ]
)
def dict_for_test_expected_value_with_compound_observable(request):
    return request.param


# 3粒子Qubit系Qubitに対する観測量、観測対象Qubits、ランダムシードのfixture
@pytest.fixture(
    params=[
        {
            "observable": Observable(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ObservedBasis(
                    [
                        Qubits([[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 1 + 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 1 + 0j]], [[0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[1 + 0j, 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [1 + 0j, 0j]]]),
                        Qubits([[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1 + 0j]]]),
                    ]
                ),
            ),
            "qubits": Qubits(
                [
                    [[sqrt(0.25) + 0j, 0j], [0j, 0j]],
                    [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
                ]
            ),
            "randomize_seed": 1,
        },
    ]
)
def dict_for_test_observation_with_compound_observable(request):
    return request.param


# 2粒子Qubit系に対する観測量の組および結合後の観測量のfixture
@pytest.fixture(
    params=[
        {
            "observable_group": [
                Observable(
                    [1, 2, 3, 4],
                    ObservedBasis(
                        [
                            Qubits([[1 + 0j, 0j], [0j, 0j]]),
                            Qubits([[0j, 1 + 0j], [0j, 0j]]),
                            Qubits([[0j, 0j], [1 + 0j, 0j]]),
                            Qubits([[0j, 0j], [0j, 1 + 0j]]),
                        ]
                    ),
                ),
                Observable(
                    [100, -100, 1000, -1000],
                    ObservedBasis(
                        [
                            Qubits([[1 + 0j, 0j], [0j, 0j]]),
                            Qubits([[0j, 1 + 0j], [0j, 0j]]),
                            Qubits([[0j, 0j], [1 + 0j, 0j]]),
                            Qubits([[0j, 0j], [0j, 1 + 0j]]),
                        ]
                    ),
                ),
            ],
            "expected_matrix": [
                [100 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, -100 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 1000 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    -1000 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ],
                [0j, 0j, 0j, 0j, 200 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, -200 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 2000 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    -2000 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 300 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, -300 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 3000 + 0j, 0j, 0j, 0j, 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    -3000 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 400 + 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, -400 + 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 4000 + 0j, 0j],
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    -4000 + 0j,
                ],
            ],
        }
    ]
)
def dict_for_test_combine_observables(request):
    return request.param
