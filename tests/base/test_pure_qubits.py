from test.support import captured_stdout

import numpy
import pytest

from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    QubitCountNotMatchError,
    EmptyArgsError,
)
from quantum_simulator.base.pure_qubits import (
    OrthogonalSystem,
    PureQubits,
    _is_not_pure_qubits,
    all_orthogonal,
    combine,
    combine_ons,
    inner,
    is_orthogonal,
    multiple_combine,
    multiple_combine_ons,
)
from quantum_simulator.base.utils import allclose


class TestPureQubits:
    """
    PureQubitsクラスと付随するメソッドのテスト
    """

    def test_for_true__is_not_pure_qubits(self, invalid_pure_qubits_amp):
        """_is_not_pure_qubitsメソッドが真になる正常系テスト"""

        array = numpy.array(invalid_pure_qubits_amp)
        assert _is_not_pure_qubits(array)

    def test_for_false__is_not_pure_qubits(self, valid_pure_qubits_amp):
        """_is_not_pure_qubitsメソッドが偽になる正常系テスト"""

        array = numpy.array(valid_pure_qubits_amp["amplitudes"])
        assert not _is_not_pure_qubits(array)

    def test_for_success_consructor(self, valid_pure_qubits_amp):
        """
        __init__メソッドの正常系テスト
        """
        qubits_object = PureQubits(valid_pure_qubits_amp["amplitudes"])
        assert valid_pure_qubits_amp["qubits_count"] == qubits_object.qubit_count

    def test_for_success_dirac_notation(self, valid_pure_qubits_amp):
        """
        dirac_notationメソッドの正常系テスト
        """
        # dirac_notationメソッドのテスト
        qubits_object = PureQubits(valid_pure_qubits_amp["amplitudes"])
        dirac_notation = valid_pure_qubits_amp["dirac_notation"]
        with captured_stdout() as stdout:
            qubits_object.dirac_notation()
            assert stdout.getvalue() == dirac_notation

    def test_for_failure_constructor(self, invalid_pure_qubits_amp):
        """
        __init__メソッドの異常系テスト
        """
        with pytest.raises(InitializeError):
            PureQubits(invalid_pure_qubits_amp)

#     def test_for_combine(self, pair_pure_qubits):
#         """
#         combineメソッドのテスト
#         """
#         qubits_0 = pair_pure_qubits["qubits_0"]
#         qubits_1 = pair_pure_qubits["qubits_1"]
#         if qubits_0 is None and qubits_1 is None:
#             with pytest.raises(EmptyArgsError):
#                 combine(qubits_0, qubits_1)
#         else:
#             combined_qubits = combine(qubits_0, qubits_1)
#             expected_combined_qubits = pair_pure_qubits["combined_qubits"]
#             assert allclose(combined_qubits.vector, expected_combined_qubits.vector)

#     def test_for_multiple_combine(self, list_pure_qubits):
#         """
#         multiple_combineメソッドのテスト
#         """
#         qubits_list = list_pure_qubits["qubits_list"]
#         if qubits_list == []:
#             with pytest.raises(EmptyArgsError):
#                 multiple_combine(qubits_list)
#         combined_qubits = multiple_combine(qubits_list)
#         expected_combined_qubits = list_pure_qubits["combined_qubits"]
#         assert allclose(combined_qubits.vector, expected_combined_qubits.vector)

#     def test_for_success_inner(self, dict_for_test_valid_inner_input):
#         """
#         innerメソッドの正常系テスト
#         """
#         target_0 = dict_for_test_valid_inner_input["target_0"]
#         target_1 = dict_for_test_valid_inner_input["target_1"]
#         result = inner(target_0, target_1)

#         expected_result = dict_for_test_valid_inner_input["result"]
#         assert allclose(result, expected_result)

#     def test_for_failure_inner(self, dict_for_test_invalid_inner_input):
#         """
#         innerメソッドの異常系テスト
#         """
#         with pytest.raises(QubitCountNotMatchError):
#             target_0 = dict_for_test_invalid_inner_input["target_0"]
#             target_1 = dict_for_test_invalid_inner_input["target_1"]
#             inner(target_0, target_1)

#     def test_for_success_is_orthogonal(self, dict_for_test_is_orthogonal):
#         """
#         is_orthogonalメソッドの正常系テスト
#         """
#         target_0 = dict_for_test_is_orthogonal["target_0"]
#         target_1 = dict_for_test_is_orthogonal["target_1"]
#         result = is_orthogonal(target_0, target_1)

#         expected_result = dict_for_test_is_orthogonal["result"]
#         assert result == expected_result

#     def test_for_success_all_orthogonal(self, dict_for_test_all_orthogonal):
#         """
#         all_orthogonalメソッドの正常系テスト
#         """
#         target = dict_for_test_all_orthogonal["target"]
#         result = all_orthogonal(target)

#         expected_result = dict_for_test_all_orthogonal["result"]
#         assert result == expected_result

#     def test_for_failure_all_orthogonal(self):
#         """
#         all_orthogonalメソッドの異常系テスト
#         """
#         with pytest.raises(NoQubitsInputError):
#             all_orthogonal([])


# class TestOrthogonalSystem:
#     """
#     OrthogonalSystemクラスと付随するメソッドのテスト
#         * 以下のロジックはテストしない
#             * ただ値を代入するだけのロジック
#             * すでにテスト済みの値を代入するロジック
#             * 既存オブジェクトを出力するだけのロジック
#     """

#     def test_for_success_onb_constructor(self, dict_for_test_success_onb_constructor):
#         """
#         __init__メソッドの正常系テスト
#         """
#         qubits_list = dict_for_test_success_onb_constructor
#         onb = OrthogonalSystem(qubits_list)
#         assert onb.qubits_list == qubits_list

#     def test_for_non_orthogonal_onb_constructor(
#         self, dict_for_test_non_orthogonal_onb_constructor
#     ):
#         """
#         __init__メソッドの異常系テスト
#         (非直交)
#         """
#         with pytest.raises(InitializeError) as error:
#             qubits_list = dict_for_test_non_orthogonal_onb_constructor
#             OrthogonalSystem(qubits_list)
#         assert "与えられたQubit群のリストは互いに直交しません" in str(error.value)

#     def test_for_success_is_onb(self, dict_for_test_success_onb):
#         """
#         onbメソッドの正常系テスト (True)
#         """
#         qubits_list = dict_for_test_success_onb
#         ons = OrthogonalSystem(qubits_list)
#         assert ons.is_onb()

#     def test_for_failure_is_onb(self, dict_for_test_failure_onb):
#         """
#         onbメソッドの正常系テスト (False)
#         """
#         qubits_list = dict_for_test_failure_onb
#         ons = OrthogonalSystem(qubits_list)
#         assert not ons.is_onb()

#     def test_for_combine_ons(self, dict_for_test_combine_ons):
#         """combin_onsメソッドの異常系テスト"""
#         ons_0 = OrthogonalSystem(dict_for_test_combine_ons["ons_0"])
#         ons_1 = OrthogonalSystem(dict_for_test_combine_ons["ons_1"])
#         expected_result = OrthogonalSystem(dict_for_test_combine_ons["result"])

#         result = combine_ons(ons_0, ons_1)

#         assert len(result.qubits_list) == len(expected_result.qubits_list)
#         for index in range(len(result.qubits_list)):
#             qubits_0 = result.qubits_list[index]
#             qubits_1 = expected_result.qubits_list[index]
#             allclose(qubits_0.vector, qubits_1.vector)

#     def test_for_multiple_combine_ons(self, dict_for_test_multiple_combine_ons):
#         """multiple_combin_onsメソッドの異常系テスト"""
#         ons_list = dict_for_test_multiple_combine_ons["ons_list"]
#         expected_result = OrthogonalSystem(dict_for_test_multiple_combine_ons["result"])

#         result = multiple_combine_ons(ons_list)

#         assert len(result.qubits_list) == len(expected_result.qubits_list)
#         for index in range(len(result.qubits_list)):
#             qubits_0 = result.qubits_list[index]
#             qubits_1 = expected_result.qubits_list[index]
#             allclose(qubits_0.vector, qubits_1.vector)
