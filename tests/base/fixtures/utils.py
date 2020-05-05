import pytest


@pytest.fixture(params=[0, -2, -128, 1, 10, 100])
def not_pow2_int(request):
    """is_pow2メソッド 返り値Falseのテスト用fixture"""
    return request.param


@pytest.fixture(params=[2, 4, 8, 16, 64, 128, 256])
def pow2_int(request):
    """is_pow2メソッド 返り値Trueのテスト用fixture"""
    return request.param
