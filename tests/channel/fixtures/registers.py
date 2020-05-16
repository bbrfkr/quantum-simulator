import pytest


@pytest.fixture(params=[0, 1, 9])
def length_registers(request):
    """レジスタの長さを返すfixture"""
    return request.param


@pytest.fixture(
    params=[
        {"count": 3, "insert_index": 2, "value": -0.3},
        {"count": 9, "insert_index": 0, "value": 100.0},
    ]
)
def dict_for_test_success_registers_put(request):
    """putメソッドの正常系テスト用fixture"""
    return request.param


@pytest.fixture(
    params=[
        {"count": 3, "insert_index": 2, "value": -0.3},
        {"count": 9, "insert_index": 0, "value": 100.0},
    ]
)
def dict_for_test_success_registers_get(request):
    """getメソッドの正常系テスト用fixture"""
    return request.param
