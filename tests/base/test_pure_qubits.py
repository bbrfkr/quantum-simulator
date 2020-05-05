from test.support import captured_stdout

import numpy as np
import pytest

from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    QubitCountNotMatchError,
)
from quantum_simulator.base.pure_qubits import (  # combine,; inner,; is_all_orthogonal,; is_orthogonal,
    PureQubits,
    count_qubits,
    is_pure_qubits,
    resolve_arrays,
    combine,
    inner,
    is_orthogonal,
    all_orthogonal
)
from quantum_simulator.base.utils import allclose, isclose


class TestPureQubits:
    """
    PureQubitsクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_is_pure_qubits(self, valid_pure_qubits_amp):
        """is_pure_qubitsメソッドの正常系テスト"""

        array = np.array(valid_pure_qubits_amp)
        assert is_pure_qubits(array)

    def test_for_failure_is_pure_qubits(self, invalid_pure_qubits_amp):
        """is_pure_qubitsメソッドの異常系テスト"""

        array = np.array(invalid_pure_qubits_amp)
        assert not is_pure_qubits(array)

    def test_for_success_count_qubits(self, dict_for_test_count_qubits):
        """count_qubitsメソッドの正常系テスト"""

        array = np.array(dict_for_test_count_qubits["array"])
        assert count_qubits(array) == dict_for_test_count_qubits["count"]

    def test_for_success_resolve_arrays(self, dict_for_test_resolve_arrays):
        """resolve_arraysメソッドの正常系テスト"""

        target = np.array(dict_for_test_resolve_arrays["target"])
        vector, array = resolve_arrays(target)
        expected_vector = np.array(dict_for_test_resolve_arrays["vector"])
        expected_array = np.array(dict_for_test_resolve_arrays["array"])
        assert np.all(vector == expected_vector)
        assert vector.shape == expected_vector.shape
        assert np.all(array == expected_array)
        assert array.shape == expected_array.shape

    def test_for_success_constructor(self, dict_for_test_constructor):
        """
        __init__メソッドの正常系テスト
        """
        qubits = PureQubits(dict_for_test_constructor["target"])
        assert allclose(qubits.projection, dict_for_test_constructor["projection"])
        assert allclose(
            qubits.projection_matrix, dict_for_test_constructor["projection_matrix"]
        )
        assert (
            qubits.projection_matrix_dim
            == dict_for_test_constructor["projection_matrix_dim"]
        )
        with captured_stdout() as stdout:
            qubits.dirac_notation()
        assert stdout.getvalue() == dict_for_test_constructor["dirac_notation"]

    def test_for_failure_constructor(self, invalid_pure_qubits_amp):
        """
        __init__メソッドの異常系テスト
        """
        with pytest.raises(InitializeError):
            PureQubits(invalid_pure_qubits_amp)       
        
    def test_for_success_combine(self, dict_for_test_combine):
        """
        combineメソッドの正常系テスト
        """
        target_0 = PureQubits(dict_for_test_combine["target_0"])
        target_1 = PureQubits(dict_for_test_combine["target_1"])
        result = combine(target_0, target_1)

        expected_result = PureQubits(dict_for_test_combine["result"])
        assert allclose(result.array, expected_result.array)
        assert allclose(result.vector, expected_result.vector)
        assert result.qubit_count == expected_result.qubit_count
        assert allclose(result.projection, expected_result.projection)
        assert allclose(result.projection_matrix, expected_result.projection_matrix)
        assert result.projection_matrix_dim == expected_result.projection_matrix_dim

    def test_for_success_inner(self, dict_for_test_valid_inner_input):
        """
        innerメソッドの異常系テスト
        """
        target_0 = dict_for_test_valid_inner_input["target_0"]
        target_1 = dict_for_test_valid_inner_input["target_1"]
        result = inner(target_0, target_1)
        
        expected_result = dict_for_test_valid_inner_input["result"]
        assert isclose(result, expected_result)

    def test_for_failure_inner(self, dict_for_test_invalid_inner_input):
        """
        innerメソッドの異常系テスト
        """
        with pytest.raises(QubitCountNotMatchError):
            target_0 = dict_for_test_invalid_inner_input["target_0"]
            target_1 = dict_for_test_invalid_inner_input["target_1"]
            inner(target_0, target_1)

    def test_for_success_is_orthogonal(self, dict_for_test_is_orthogonal):
        """
        is_orthogonalメソッドの正常系テスト
        """
        target_0 = dict_for_test_is_orthogonal["target_0"]
        target_1 = dict_for_test_is_orthogonal["target_1"]
        result = is_orthogonal(target_0, target_1)

        expected_result = dict_for_test_is_orthogonal["result"]
        assert result == expected_result

    def test_for_success_all_orthogonal(self, dict_for_test_all_orthogonal):
        """
        all_orthogonalメソッドの正常系テスト
        """
        target = dict_for_test_all_orthogonal["target"]
        result = all_orthogonal(target)

        expected_result = dict_for_test_all_orthogonal["result"]
        assert result == expected_result

    def test_for_failure_all_orthogonal(self):
        """
        all_orthogonalメソッドの異常系テスト
        """
        with pytest.raises(NoQubitsInputError):
            all_orthogonal([])

        # 正常系

    # # 単一Qubitに対するテスト
    # def test_valid_pure_qubit_input(self, valid_pure_qubit_amp):
    #     """[正常系]: 単一Qubit生成"""
    #     array = valid_pure_qubit_amp
    #     qubit = PureQubits(array)
    #     assert np.all(qubit.array == array)
    #     assert qubit.qubit_count == 1

    # def test_invalid_pure_qubits_input(self, invalid_pure_qubits_amp):
    #     """[異常系]: 不正な単一Qubitパラメータ"""
    #     with pytest.raises(InitializeError):
    #         array = invalid_pure_qubits_amp
    #         PureQubits(array)

    # def test_zero_inner_product_of_qubits(self, orthogonal_qubits):
    #     """[正常系]: 直交した単一Qubit同士の内積"""
    #     assert is_orthogonal(orthogonal_qubits[0], orthogonal_qubits[1])

    # def test_non_zero_inner_product_of_qubits(self, non_orthogonal_qubits):
    #     """[正常系]: 直交しない単一Qubit同士の内積"""
    #     assert not is_orthogonal(non_orthogonal_qubits[0], non_orthogonal_qubits[1])

    # def test_one_inner_product_of_qubits(self, valid_qubit):
    #     """[正常系]: 長さが1の単一Qubitに対する内積"""
    #     assert round(inner(valid_qubit, valid_qubit) - 1.0, APPROX_DIGIT) == 0.0

    # def test_matrix_of_one_qubit(self, proj_for_valid_qubit):
    #     """[正常系]: 単一Qubitに対する射影作用素"""
    #     assert np.all(
    #         np.round(
    #             proj_for_valid_qubit["qubit"].projection
    #             - np.array(proj_for_valid_qubit["projection"]),
    #             APPROX_DIGIT,
    #         )
    #         == 0.0
    #     )

    # # Qubit群に対するテスト
    # def test_valid_pure_qubits_input(self, valid_pure_qubits_amp):
    #     """[正常系]: Qubit群生成"""
    #     array = valid_pure_qubits_amp
    #     qubit = PureQubits(array)
    #     assert np.all(qubit.array == array)
    #     assert qubit.qubit_count == len(np.array(array).shape)

    # def test_combine_qubits(self, dict_test_for_combine):
    #     """[正常系]: Qubit群同士の結合"""
    #     qubits_group = dict_test_for_combine["qubits_group"]
    #     combined_qubits = combine(qubits_group[0], qubits_group[1])
    #     expected_result = np.array(dict_test_for_combine["result"])
    #     assert np.all(
    #         np.round(abs(combined_qubits.array - expected_result), APPROX_DIGIT) == 0.0
    #     )

    # def test_zero_inner_product_of_two_qubits(self, orthogonal_two_pure_qubits_groups):
    #     """[正常系]: 直交した二つのQubit群の内積"""
    #     assert (
    #         round(
    #             inner(
    #                 orthogonal_two_pure_qubits_groups[0],
    #                 orthogonal_two_pure_qubits_groups[1],
    #             ),
    #             APPROX_DIGIT,
    #         )
    #         == 0.0
    #     )
    #     assert is_orthogonal(
    #         orthogonal_two_pure_qubits_groups[0], orthogonal_two_pure_qubits_groups[1]
    #     )

    # def test_one_inner_product_of_two_qubits(self, valid_qubits):
    #     """[正常系]: 長さが1のQubit群に対する内積"""
    #     assert round(inner(valid_qubits, valid_qubits) - 1.0, APPROX_DIGIT) == 0.0

    # def test_non_zero_inner_product_of_two_qubits(
    #     self, non_orthogonal_two_pure_qubits_groups
    # ):
    #     """[正常系]: 直交しない二つのQubit群同士の内積"""
    #     assert not is_orthogonal(
    #         non_orthogonal_two_pure_qubits_groups[0],
    #         non_orthogonal_two_pure_qubits_groups[1],
    #     )

    # def test_inner_product_of_two_pure_qubits_with_unmatch_counts(
    #     self, not_match_counts_two_pure_qubits_groups
    # ):
    #     """[異常系]: 異なるQubit数の二つのQubit群に対する内積"""
    #     with pytest.raises(QubitCountNotMatchError):
    #         inner(
    #             not_match_counts_two_pure_qubits_groups[0],
    #             not_match_counts_two_pure_qubits_groups[1],
    #         )

    # def test_check_of_orthogonality_for_empty_set(self):
    #     """[異常系]: 空集合に対する直交性のテスト"""
    #     with pytest.raises(NoQubitsInputError):
    #         is_all_orthogonal([])

    # def test_check_of_orthogonality_for_unit_set(self, valid_qubits):
    #     """[正常系]: 単一Qubit群に対する直交性のテスト"""
    #     assert is_all_orthogonal([valid_qubits])

    # def test_check_of_orthogonality_for_multiple_pure_qubits_groups(
    #     self, orthogonal_multiple_pure_qubits_groups,
    # ):
    #     """[正常系]: 互いに直交する二つ以上のQubit群に対する直交性のテスト"""
    #     assert is_all_orthogonal(orthogonal_multiple_pure_qubits_groups)

    # def test_check_of_non_orthogonality_for_multiple_pure_qubits_groups(
    #     self, non_orthogonal_multiple_pure_qubits_groups
    # ):
    #     """[正常系]: 互いに直交しない二つ以上のQubit群に対する直交性のテスト"""
    #     assert not is_all_orthogonal(non_orthogonal_multiple_pure_qubits_groups)

    # def test_matrix_of_multiple_qubits(self, proj_for_valid_qubits):
    #     """[正常系]: 単一Qubitに対する射影作用素"""
    #     assert np.all(
    #         np.round(
    #             proj_for_valid_qubits["qubits"].projection
    #             - np.array(proj_for_valid_qubits["projection"]),
    #             APPROX_DIGIT,
    #         )
    #         == 0.0
    #     )
