from math import sqrt

import pytest

from base.conf import approx_digit
from base.error import InitializeError
from base.qubit import Qubit, inner, is_orthogonal


# 妥当なqubitのfixture
@pytest.fixture(
    params=[
        [1 + 0j, 0 + 0j],
        [0 + 0j, 1 + 0j],
        [sqrt(0.5) + 0j, sqrt(0.5) + 0j],
        [0 + sqrt(0.5) * 1j, 0 + sqrt(0.5) * 1j],
        [sqrt(0.4) + 0j, 0 + sqrt(0.6) * 1j],
    ]
)
def valid_qubit(request):
    return request.param


# 不正なqubitのfixture
@pytest.fixture(
    params=[
        [0 + 0j, 0 + 0j],
        [sqrt(0.3) + 0j, sqrt(0.3) + 0j],
        [0 + sqrt(0.3) * 1j, 0 + sqrt(0.3) * 1j],
        [sqrt(0.6) + 0j, sqrt(0.6) + 0j],
        [0 + sqrt(0.6) * 1j, 0 + sqrt(0.6) * 1j],
    ]
)
def invalid_qubit(request):
    return request.param


# 直交するqubit同士のfixture
@pytest.fixture(
    params=[
        [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]],
        [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [0 + sqrt(0.5) * 1j, 0 - sqrt(0.5) * 1j]],
        [[sqrt(0.4) + 0j, 0 + sqrt(0.6) * 1j], [sqrt(0.6) + 0j, 0 - sqrt(0.4) * 1j]],
    ]
)
def orthogonal_qubits(request):
    qubits = [Qubit(amplitudes[0], amplitudes[1]) for amplitudes in request.param]
    return qubits


class TestQubit:
    def test_valid_qubit_input(self, valid_qubit):
        """[正常系]: Qubit生成"""
        qubit = Qubit(valid_qubit[0], valid_qubit[1])
        assert qubit.amplitudes[0] == valid_qubit[0]
        assert qubit.amplitudes[1] == valid_qubit[1]

    def test_invalid_qubit_input(self, invalid_qubit):
        """[異常系]: 不正なQubitパラメータ"""
        with pytest.raises(InitializeError):
            Qubit(invalid_qubit[0], invalid_qubit[1])

    def test_zero_inner_product_of_qubits(self, orthogonal_qubits):
        """[正常系]: 直交したQubit同士の内積"""
        assert is_orthogonal(orthogonal_qubits[0], orthogonal_qubits[1])

    def test_one_inner_product_of_qubits(self, valid_qubit):

        qubit_0 = Qubit(valid_qubit[0], valid_qubit[1])
        qubit_1 = Qubit(valid_qubit[0], valid_qubit[1])
        assert round(inner(qubit_0, qubit_1), approx_digit) == 1.0
