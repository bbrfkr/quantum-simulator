from math import sqrt

import pytest

from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits


@pytest.fixture(
    params=[
        # ベクトル形式
        {
            "amplitudes": [1 + 0j, 0j],
            "qubits_count": 1,
            "dirac_notation": ("(1+0j)|0> +\n0j|1>\n"),
        },
        {
            "amplitudes": [0j, 0j, 1 + 0j, 0j],
            "qubits_count": 2,
            "dirac_notation": ("0j|00> +\n0j|01> +\n(1+0j)|10> +\n0j|11>\n"),
        },
        {
            "amplitudes": [1 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            "qubits_count": 3,
            "dirac_notation": (
                "(1+0j)|000> +\n0j|001> +\n0j|010> +\n0j|011> +\n"
                "0j|100> +\n0j|101> +\n0j|110> +\n0j|111>\n"
            ),
        },
        {
            "amplitudes": [0j, 0j, 0j, 0j, 0j, 0j, 0j, 1j],
            "qubits_count": 3,
            "dirac_notation": (
                "0j|000> +\n0j|001> +\n0j|010> +\n0j|011> +\n"
                "0j|100> +\n0j|101> +\n0j|110> +\n1j|111>\n"
            ),
        },
        {
            "amplitudes": [
                sqrt(2 / 16) * 1j,
                sqrt(1 / 16) + 0j,
                sqrt(3 / 16) * 1j,
                sqrt(1 / 16) + 0j,
                sqrt(1 / 16) * 1j,
                sqrt(5 / 16) + 0j,
                sqrt(1 / 16) * 1j,
                sqrt(2 / 16) + 0j,
            ],
            "qubits_count": 3,
            "dirac_notation": None,
        },
        {
            "amplitudes": [
                sqrt(0.25) + 0j,
                0j,
                0j,
                sqrt(0.25) + 0j,
                sqrt(0.25) + 0j,
                0j,
                0j,
                sqrt(0.25) + 0j,
            ],
            "qubits_count": 3,
            "dirac_notation": "(0.5+0j)|000> +\n0j|001> +\n0j|010> +\n(0.5+0j)|011> +\n"
            "(0.5+0j)|100> +\n0j|101> +\n0j|110> +\n(0.5+0j)|111>\n",
        },
        # 2qubits以上のndarray形式
        {
            "amplitudes": [[[1 + 0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 0j]]],
            "qubits_count": 3,
            "dirac_notation": (
                "(1+0j)|000> +\n0j|001> +\n0j|010> +\n0j|011> +\n"
                "0j|100> +\n0j|101> +\n0j|110> +\n0j|111>\n"
            ),
        },
        {
            "amplitudes": [[[0j, 0j], [0j, 0j]], [[0j, 0j], [0j, 1j]]],
            "qubits_count": 3,
            "dirac_notation": (
                "0j|000> +\n0j|001> +\n0j|010> +\n0j|011> +\n"
                "0j|100> +\n0j|101> +\n0j|110> +\n1j|111>\n"
            ),
        },
        {
            "amplitudes": [
                [[sqrt(0.25) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
                [[sqrt(0.25) + 0j, 0j], [0j, sqrt(0.25) + 0j]],
            ],
            "qubits_count": 3,
            "dirac_notation": "(0.5+0j)|000> +\n0j|001> +\n0j|010> +\n(0.5+0j)|011> +\n"
            "(0.5+0j)|100> +\n0j|101> +\n0j|110> +\n(0.5+0j)|111>\n",
        },
        {
            "amplitudes": [
                [
                    [sqrt(2 / 16) * 1j, sqrt(1 / 16) + 0j],
                    [sqrt(3 / 16) * 1j, sqrt(1 / 16) + 0j],
                ],
                [
                    [sqrt(1 / 16) * 1j, sqrt(5 / 16) + 0j],
                    [sqrt(1 / 16) * 1j, sqrt(2 / 16) + 0j],
                ],
            ],
            "qubits_count": 3,
            "dirac_notation": None,
        },
    ]
)
def valid_pure_qubits_amp(request):
    """妥当なPureQubitsに対する確率振幅のfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 空
        [],
        # 空間が(C^2)^nとならない、ベクトル
        [0j, 1 + 0j, 0j],
        [0j, 0j, 0j, 0j, 0j, 1 + 0j, 0j],
        # 空間が(C^2)^nとならない、ndarray
        [
            [
                [sqrt(2 / 16) * 1j, sqrt(1 / 16) + 0j],
                [sqrt(3 / 16) * 1j, sqrt(1 / 16) + 0j],
            ],
            [
                [sqrt(1 / 16) * 1j, sqrt(5 / 16) + 0j],
                [sqrt(1 / 16) * 1j, sqrt(2 / 16) + 0j],
            ],
            [[0j, 0j], [0j, 0j]],
        ],
        # 長さが1ではない
        [sqrt(0.3) * 1j, sqrt(0.6) + 0j],
        [[sqrt(0.5) + 0j, 0j], [sqrt(0.5) * 1j, sqrt(0.1) + 0j]],
    ]
)
def invalid_pure_qubits_amp(request):
    """妥当でないPureQubitsに対する確率振幅のfixture"""
    return request.param


@pytest.fixture(
    params=[
        {"qubits_0": None, "qubits_1": None, "combined_qubits": None},
        {
            "qubits_0": PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j]),
            "qubits_1": None,
            "combined_qubits": PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j]),
        },
        {
            "qubits_0": None,
            "qubits_1": PureQubits([sqrt(0.5) + 0j, 0j, sqrt(0.5) + 0j, 0j]),
            "combined_qubits": PureQubits([sqrt(0.5) + 0j, 0j, sqrt(0.5) + 0j, 0j]),
        },
        {
            "qubits_0": PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j]),
            "qubits_1": PureQubits([sqrt(0.5) + 0j, 0j, sqrt(0.5) + 0j, 0j]),
            "combined_qubits": PureQubits(
                [
                    0.5 + 0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ]
            ),
        },
    ]
)
def pair_pure_qubits(request):
    """純粋状態のペアのfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "qubits_list": [
                PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([0j, 1.0 + 0j]),
                PureQubits([1.0 + 0j, 0j]),
            ],
            "combined_qubits": PureQubits(
                [0j, 0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j, 0j, 0j, 0j]
            ),
        },
    ]
)
def list_pure_qubits(request):
    """純粋状態のリストのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 1 : 2 Qubits同士
        {
            "target_0": PureQubits([1 + 0j, 0j]),
            "target_1": PureQubits([0j, 1 + 0j, 0j, 0j]),
        },
        # 4 : 2 Qubits同士
        {
            "target_0": PureQubits(
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ]
            ),
            "target_1": PureQubits([sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]),
        },
    ]
)
def invalid_inner_input_qubits(request):
    """innerメソッドテスト用の不正なインプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 単一Qubit同士
        {
            "target_0": PureQubits([0 + 1j, 0j]),
            "target_1": PureQubits([1 + 0j, 0j]),
            "result": -1j,
        },
        {
            "target_0": PureQubits(
                [0j, sqrt(0.5) + 0j, sqrt(0.25) + 0j, sqrt(0.25) + 0j]
            ),
            "target_1": PureQubits(
                [sqrt(0.5) + 0j, 0j, sqrt(0.25) + 0j, sqrt(0.25) * 1j]
            ),
            "result": 0.25 + 0.25j,
        },
    ]
)
def dict_for_test_valid_inner_input(request):
    """innerメソッドテスト用の妥当なインプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 1 : 2 Qubits同士
        {
            "target_0": PureQubits([1 + 0j, 0j]),
            "target_1": PureQubits([0j, 1 + 0j, 0j, 0j]),
        },
        # 4 : 2 Qubits同士
        {
            "target_0": PureQubits(
                [
                    0j,
                    0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0.5 + 0j,
                    0j,
                    0j,
                    0j,
                    0j,
                ]
            ),
            "target_1": PureQubits([sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]),
        },
    ]
)
def dict_for_test_invalid_inner_input(request):
    """innerメソッドテスト用の不正なインプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 直交を期待する場合
        {
            "target_0": PureQubits([1 + 0j, 0j]),
            "target_1": PureQubits([0j, 1 + 0j]),
            "result": True,
        },
        {
            "target_0": PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
            "target_1": PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            "result": True,
        },
        {
            "target_0": PureQubits(
                [[sqrt(0.25) + 0j, sqrt(0.25) + 0j], [sqrt(0.25) + 0j, sqrt(0.25) + 0j]]
            ),
            "target_1": PureQubits(
                [
                    [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                    [-sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                ]
            ),
            "result": True,
        },
        # 直交を期待しない場合
        {
            "target_0": PureQubits([0 + 1j, 0j]),
            "target_1": PureQubits([1 + 0j, 0j]),
            "result": False,
        },
        {
            "target_0": PureQubits([sqrt(0.5) + 0j, sqrt(0.25) + sqrt(0.25) * 1j]),
            "target_1": PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            "result": False,
        },
        {
            "target_0": PureQubits(
                [0j, sqrt(0.5) + 0j, sqrt(0.25) + 0j, sqrt(0.25) + 0j]
            ),
            "target_1": PureQubits(
                [sqrt(0.5) + 0j, 0j, sqrt(0.25) + 0j, sqrt(0.25) * 1j]
            ),
            "result": False,
        },
    ]
)
def dict_for_test_is_orthogonal(request):
    """is_orthogonalメソッドテスト用インプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        # 直交を期待する場合 - 要素が一つ
        {"target": [PureQubits([1 + 0j, 0j])], "result": True},
        {
            "target": [
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                    ]
                )
            ],
            "result": True,
        },
        # 直交を期待する場合 - 要素が3つ以上
        {
            "target": [
                PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
                PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
            ],
            "result": True,
        },
        {
            "target": [
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [-sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [-sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
            ],
            "result": True,
        },
        # 直交を期待しない場合
        {
            "target": [PureQubits([0 + 1j, 0j]), PureQubits([1 + 0j, 0j])],
            "result": False,
        },
        {
            "target": [
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [-sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [-sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                    ]
                ),
            ],
            "result": False,
        },
        {
            "target": [
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [-sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits(
                    [
                        [sqrt(0.25) + 0j, sqrt(0.25) + 0j],
                        [sqrt(0.25) + 0j, -sqrt(0.25) + 0j],
                    ]
                ),
                PureQubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
            ],
            "result": False,
        },
    ]
)
def dict_for_test_all_orthogonal(request):
    """all_orthogonalメソッドテスト用インプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        [PureQubits([0j, 1 + 0j]), PureQubits([1 + 0j, 0j])],
        [
            PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]),
            PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j]),
        ],
    ]
)
def dict_for_test_success_onb_constructor(request):
    """OrthogonalSystemの__init__メソッドテスト用の妥当なインプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        [PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j]), PureQubits([1 + 0j, 0j])],
        [PureQubits([1 + 0j, 0j]), PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j])],
    ]
)
def dict_for_test_non_orthogonal_onb_constructor(request):
    """OrthogonalSystemの__init__メソッドテスト用の非直交なインプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        [PureQubits([1 + 0j, 0j]), PureQubits([0j, 1 + 0j])],
        [
            PureQubits([1 + 0j, 0j, 0j, 0j]),
            PureQubits([0j, 1 + 0j, 0j, 0j]),
            PureQubits([0j, 0j, 1 + 0j, 0j]),
            PureQubits([0j, 0j, 0j, 1 + 0j]),
        ],
    ]
)
def dict_for_test_success_onb(request):
    """is_onbメソッドテスト用の充足インプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        [PureQubits([1 + 0j, 0j])],
        [
            PureQubits([1 + 0j, 0j, 0j, 0j]),
            PureQubits([0j, 1 + 0j, 0j, 0j]),
            PureQubits([0j, 0j, 1 + 0j, 0j]),
        ],
    ]
)
def dict_for_test_failure_onb(request):
    """is_onbメソッドテスト用の不足インプットのfixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "ons_0": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
            "ons_1": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
            "result": [
                PureQubits([1.0 + 0j, 0j, 0j, 0j]),
                PureQubits([0j, 1.0 + 0j, 0j, 0j]),
                PureQubits([0j, 0j, 1.0 + 0j, 0j]),
                PureQubits([0j, 0j, 0j, 1.0 + 0j]),
            ],
        },
        {
            "ons_0": [PureQubits([0j, 1.0 + 0j]), PureQubits([1.0 + 0j, 0j])],
            "ons_1": [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])],
            "result": [
                PureQubits([0j, 0j, 1.0 + 0j, 0j]),
                PureQubits([0j, 0j, 0j, 1.0 + 0j]),
                PureQubits([1.0 + 0j, 0j, 0j, 0j]),
                PureQubits([0j, 1.0 + 0j, 0j, 0j]),
            ],
        },
        {
            "ons_0": [PureQubits([0j, 1.0 + 0j]), PureQubits([1.0 + 0j, 0j])],
            "ons_1": [PureQubits([0j, 1.0 + 0j]), PureQubits([1.0 + 0j, 0j])],
            "result": [
                PureQubits([1.0 + 0j, 0j, 0j, 0j]),
                PureQubits([0j, 1.0 + 0j, 0j, 0j]),
                PureQubits([0j, 0j, 1.0 + 0j, 0j]),
                PureQubits([0j, 0j, 0j, 1.0 + 0j]),
            ],
        },
    ]
)
def dict_for_test_combine_ons(request):
    """combine_onsメソッドのテスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {
            "ons_list": [
                OrthogonalSystem(
                    [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
                ),
                OrthogonalSystem(
                    [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
                ),
                OrthogonalSystem(
                    [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
                ),
            ],
            "result": [
                PureQubits([1.0 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j]),
                PureQubits([0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j, 0j]),
                PureQubits([0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j]),
                PureQubits([0j, 0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j]),
                PureQubits([0j, 0j, 0j, 0j, 1.0 + 0j, 0j, 0j, 0j]),
                PureQubits([0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j, 0j]),
                PureQubits([0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j]),
                PureQubits([0j, 0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j]),
            ],
        },
    ]
)
def dict_for_test_multiple_combine_ons(request):
    """combine_onsメソッドのテスト用fixture"""
    return request.param
