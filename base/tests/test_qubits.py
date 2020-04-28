import numpy as np
import pytest

from base.conf import approx_digit
from base.error import InitializeError
from base.qubits import Qubits, combine, inner, is_orthogonal


class TestQubits:
    # 単一Qubitに対するテスト
    def test_valid_qubit_input(self, valid_qubit_amp):
        """[正常系]: 単一Qubit生成"""
        amplitudes = np.array(valid_qubit_amp)
        qubit = Qubits(amplitudes)
        assert np.all(qubit.amplitudes == amplitudes)
        assert qubit.qubit_count == 1

    def test_invalid_qubit_input(self, invalid_qubit_amp):
        """[異常系]: 不正な単一Qubitパラメータ"""
        with pytest.raises(InitializeError):
            amplitudes = np.array(invalid_qubit_amp)
            Qubits(amplitudes)

    def test_zero_inner_product_of_qubits(self, orthogonal_qubits):
        """[正常系]: 直交した単一Qubit同士の内積"""
        assert is_orthogonal(orthogonal_qubits[0], orthogonal_qubits[1])

    def test_non_zero_inner_product_of_qubits(self, non_orthogonal_qubits):
        """[正常系]: 直交しない単一Qubit同士の内積"""
        assert not is_orthogonal(non_orthogonal_qubits[0], non_orthogonal_qubits[1])

    def test_one_inner_product_of_qubits(self, valid_qubit):
        """[正常系]: 長さが1の単一Qubitに対する内積"""
        assert round(inner(valid_qubit, valid_qubit) - 1.0, approx_digit) == 0.0

    # 複数Qubitに対するテスト
    def test_valid_qubits_input(self, valid_qubits_amp):
        """[正常系]: 複数Qubit生成"""
        amplitudes = np.array(valid_qubits_amp)
        qubit = Qubits(amplitudes)
        assert np.all(qubit.amplitudes == amplitudes)
        assert qubit.qubit_count == len(amplitudes.shape)

    def test_combine_qubits(self, dict_test_for_combine):
        """[正常系]: 複数Qubit生成"""
        qubits_group = dict_test_for_combine["qubits_group"]
        combined_qubits = combine(qubits_group[0], qubits_group[1])
        expected_result = np.array(dict_test_for_combine["result"])
        assert np.all(
            np.round(abs(combined_qubits.amplitudes - expected_result), approx_digit)
            == 0.0
        )

    def test_zero_inner_product_of_multiple_qubits(self, orthogonal_multiple_qubits):
        """[正常系]: 直交した複数Qubit同士の内積"""
        assert is_orthogonal(
            orthogonal_multiple_qubits[0], orthogonal_multiple_qubits[1]
        )

    def test_non_zero_inner_product_of_multiple_qubits(
        self, non_orthogonal_multiple_qubits
    ):
        """[正常系]: 直交しない複数Qubit同士の内積"""
        assert not is_orthogonal(
            non_orthogonal_multiple_qubits[0], non_orthogonal_multiple_qubits[1]
        )

    def test_one_inner_product_of_multiple_qubits(self, valid_qubits):
        """[正常系]: 長さが1の複数Qubitに対する内積"""
        assert round(inner(valid_qubits, valid_qubits) - 1.0, approx_digit) == 0.0
