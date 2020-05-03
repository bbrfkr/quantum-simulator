import numpy as np
import pytest

from quantum_simulator.base.conf import APPROX_DIGIT
from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    QubitCountNotMatchError,
)
from quantum_simulator.base.pure_qubits import (
    PureQubits,
    combine,
    inner,
    is_all_orthogonal,
    is_orthogonal,
)


class TestPureQubits:
    """PureQubitsクラスと付随するメソッドのテスト"""

    # 単一Qubitに対するテスト
    def test_valid_qubit_input(self, valid_qubit_amp):
        """[正常系]: 単一Qubit生成"""
        amplitudes = valid_qubit_amp
        qubit = PureQubits(amplitudes)
        assert np.all(qubit.amplitudes == amplitudes)
        assert qubit.qubit_count == 1

    def test_invalid_qubit_input(self, invalid_qubit_amp):
        """[異常系]: 不正な単一Qubitパラメータ"""
        with pytest.raises(InitializeError):
            amplitudes = invalid_qubit_amp
            PureQubits(amplitudes)

    def test_zero_inner_product_of_qubits(self, orthogonal_qubits):
        """[正常系]: 直交した単一Qubit同士の内積"""
        assert is_orthogonal(orthogonal_qubits[0], orthogonal_qubits[1])

    def test_non_zero_inner_product_of_qubits(self, non_orthogonal_qubits):
        """[正常系]: 直交しない単一Qubit同士の内積"""
        assert not is_orthogonal(non_orthogonal_qubits[0], non_orthogonal_qubits[1])

    def test_one_inner_product_of_qubits(self, valid_qubit):
        """[正常系]: 長さが1の単一Qubitに対する内積"""
        assert round(inner(valid_qubit, valid_qubit) - 1.0, APPROX_DIGIT) == 0.0

    def test_projection_of_one_qubit(self, proj_for_valid_qubit):
        """[正常系]: 単一Qubitに対する射影作用素"""
        assert np.all(
            np.round(
                proj_for_valid_qubit["qubit"].projection
                - np.array(proj_for_valid_qubit["projection"]),
                APPROX_DIGIT,
            )
            == 0.0
        )

    # Qubit群に対するテスト
    def test_valid_qubits_input(self, valid_qubits_amp):
        """[正常系]: Qubit群生成"""
        amplitudes = valid_qubits_amp
        qubit = PureQubits(amplitudes)
        assert np.all(qubit.amplitudes == amplitudes)
        assert qubit.qubit_count == len(np.array(amplitudes).shape)

    def test_combine_qubits(self, dict_test_for_combine):
        """[正常系]: Qubit群同士の結合"""
        qubits_group = dict_test_for_combine["qubits_group"]
        combined_qubits = combine(qubits_group[0], qubits_group[1])
        expected_result = np.array(dict_test_for_combine["result"])
        assert np.all(
            np.round(abs(combined_qubits.amplitudes - expected_result), APPROX_DIGIT)
            == 0.0
        )

    def test_zero_inner_product_of_two_qubits(self, orthogonal_two_qubits_groups):
        """[正常系]: 直交した二つのQubit群の内積"""
        assert (
            round(
                inner(orthogonal_two_qubits_groups[0], orthogonal_two_qubits_groups[1]),
                APPROX_DIGIT,
            )
            == 0.0
        )
        assert is_orthogonal(
            orthogonal_two_qubits_groups[0], orthogonal_two_qubits_groups[1]
        )

    def test_one_inner_product_of_two_qubits(self, valid_qubits):
        """[正常系]: 長さが1のQubit群に対する内積"""
        assert round(inner(valid_qubits, valid_qubits) - 1.0, APPROX_DIGIT) == 0.0

    def test_non_zero_inner_product_of_two_qubits(
        self, non_orthogonal_two_qubits_groups
    ):
        """[正常系]: 直交しない二つのQubit群同士の内積"""
        assert not is_orthogonal(
            non_orthogonal_two_qubits_groups[0], non_orthogonal_two_qubits_groups[1]
        )

    def test_inner_product_of_two_qubits_with_unmatch_counts(
        self, not_match_counts_two_qubits_groups
    ):
        """[異常系]: 異なるQubit数の二つのQubit群に対する内積"""
        with pytest.raises(QubitCountNotMatchError):
            inner(
                not_match_counts_two_qubits_groups[0],
                not_match_counts_two_qubits_groups[1],
            )

    def test_check_of_orthogonality_for_empty_set(self):
        """[異常系]: 空集合に対する直交性のテスト"""
        with pytest.raises(NoQubitsInputError):
            is_all_orthogonal([])

    def test_check_of_orthogonality_for_unit_set(self, valid_qubits):
        """[正常系]: 単一Qubit群に対する直交性のテスト"""
        assert is_all_orthogonal([valid_qubits])

    def test_check_of_orthogonality_for_multiple_qubits_groups(
        self, orthogonal_multiple_qubits_groups,
    ):
        """[正常系]: 互いに直交する二つ以上のQubit群に対する直交性のテスト"""
        assert is_all_orthogonal(orthogonal_multiple_qubits_groups)

    def test_check_of_non_orthogonality_for_multiple_qubits_groups(
        self, non_orthogonal_multiple_qubits_groups
    ):
        """[正常系]: 互いに直交しない二つ以上のQubit群に対する直交性のテスト"""
        assert not is_all_orthogonal(non_orthogonal_multiple_qubits_groups)

    def test_projection_of_multiple_qubits(self, proj_for_valid_qubits):
        """[正常系]: 単一Qubitに対する射影作用素"""
        assert np.all(
            np.round(
                proj_for_valid_qubits["qubits"].projection
                - np.array(proj_for_valid_qubits["projection"]),
                APPROX_DIGIT,
            )
            == 0.0
        )
