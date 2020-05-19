import pytest


@pytest.fixture(params=[0, -2, -128, 1, 10, 100])
def not_pow2_int(request):
    """is_pow2メソッド 返り値Falseのテスト用fixture"""
    return request.param


@pytest.fixture(params=[2, 4, 8, 16, 64, 128, 256])
def pow2_int(request):
    """is_pow2メソッド 返り値Trueのテスト用fixture"""
    return request.param

@pytest.fixture(params=[-1])
def negative_value(request):
    """count_bitsメソッドの異常系テスト用fixture"""
    return request.param

@pytest.fixture(params=[
    {
        "value": 2,
        "result": 2
    },
    {
        "value": 127,
        "result": 7
    },
])
def dict_for_test_count_bits(request):
    """count_bitsメソッドの異常系テスト用fixture"""
    return request.param

@pytest.fixture(params=[[0.2, 0.5, 0.1, 0.2], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
def probabilities_list(request):
    """is_probabilitiesメソッド 返り値Trueのテスト用fixture"""
    return request.param


@pytest.fixture(
    params=[[0.2, 0.5, 0.1, 0.1], [0.2, 0.5, -0.1, 0.4], [0.2, 0.5, 0.3, 0.1]]
)
def not_probabilities_list(request):
    """is_probabilitiesメソッド 返り値Falseのテスト用fixture"""
    return request.param
