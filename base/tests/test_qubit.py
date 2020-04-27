import pytest

from base.conf import approx_digit
from base.error import InitializeError
from base.qubit import Qubit, inner, is_orthogonal


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
