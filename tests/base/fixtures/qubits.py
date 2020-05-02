from math import sqrt

import pytest

from quantum_simulator.base.qubits import Qubits


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
    """妥当な単一qubitに対する確率振幅のfixture"""
    return request.param


@pytest.fixture()
def valid_qubit(valid_qubit_amp):
    """妥当な単一qubitのfixture"""
    amplitudes = valid_qubit_amp
    return Qubits(amplitudes)


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
    """不正な単一qubitに対する確率振幅のfixture"""
    return request.param


@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [0j, 1 + 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, -sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.6) + 0j, -sqrt(0.4) * 1j]],
    ]
)
def orthogonal_qubits(request):
    """直交する単一qubit同士のfixture"""
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [1 + 0j, 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) * 1j, sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, sqrt(0.6) * 1j], [sqrt(0.4) + 0j, -sqrt(0.6) * 1j]],
    ]
)
def non_orthogonal_qubits(request):
    """直交しない単一qubit同士のfixture"""
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


@pytest.fixture(
    params=[
        [[1 + 0j, 0j], [0j, 0j]],
        [[[0j, 0j], [1 + 0j, 0j]], [[0j, 0j], [0j, 0j]]],
        [[[0j, 0j], [0j, 0j]], [[0j, 1 + 0j], [0j, 0j]]],
        [[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]],
    ]
)
def valid_qubits_amp(request):
    """
    妥当なQubit群に対する確率振幅のfixture
    2粒子: |00>
    3粒子: |010>
    4粒子: |1001>
    重ね合わせ2粒子(EPR pair): sqrt(0.5)|00> + sqrt(0.5)|11>
    """
    return request.param


@pytest.fixture()
def valid_qubits(valid_qubits_amp):
    """妥当なQubit群のfixture"""
    amplitudes = valid_qubits_amp
    return Qubits(amplitudes)


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
    """単一Qubitに対する射影作用素のfixture"""
    return request.param


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
    """
    Qubit群に対する射影作用素のfixture
    2粒子: |00>
    3粒子: |010>
    4粒子: |1001>
    重ね合わせ2粒子(EPR pair): sqrt(0.5)|00> + sqrt(0.5)|11>
    """
    return request.param


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
    """
    粒子間結合テスト用fixture
    test1: |0> x |+> = sqrt(0.5)|00> + sqrt(0.5)|01>
    test2: |+> x |-> = 0.5|00> - 0.5|01> + 0.5|10> - 0.5|11>
    test3: |0-> x |1> = sqrt(0.5)|001> - sqrt(0.5)|011>
    """
    test_dict = {
        "qubits_group": [
            Qubits(amplitudes) for amplitudes in request.param["qubits_group"]
        ],
        "result": request.param["result"],
    }
    return test_dict


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
    """
    直交する二つのQubit群同士のfixture
    test1: |0+>, |1+>
    test2: |0->, |0+>
    """
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


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
    """
    直交しない二つのQubit群のfixture
    test1: {|-+>, |1+>} (<1+|-+>)
    test2: {|0->, |01>} (<01|0->)
    """
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


@pytest.fixture(
    params=[
        [[[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]], [0j, 1.0 + 0j]],
        [[1.0 + 0j, 0j], [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0j, 0j]]],
    ]
)
def not_match_counts_two_qubits_groups(request):
    """
    異なるQubit数の二つのQubit群のfixture
    test1: |0+>, |1>
    test2: |0>, |0+>
    """
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


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
    """
    互いに直交する二つより多いQubit群同士のfixture
    test1: {|00>, |01>, |10>, |11>}
    """
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits


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
    """
    互いに直交しない二つより多いQubit群同士のfixture
    test1: {|00>, |01>, |10>, |11>, |00>}
    """
    amplitudes_list = [amplitudes for amplitudes in request.param]
    qubits = [Qubits(amplitudes) for amplitudes in amplitudes_list]
    return qubits
