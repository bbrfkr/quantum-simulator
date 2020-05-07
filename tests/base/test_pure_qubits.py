from test.support import captured_stdout

import numpy as np
import pytest

from quantum_simulator.base.error import (
    InitializeError,
    NoQubitsInputError,
    QubitCountNotMatchError,
)
from quantum_simulator.base.pure_qubits import (
    OrthogonalSystem,
    PureQubits,
    _count_qubits,
    _is_pure_qubits,
    _resolve_arrays,
    all_orthogonal,
    combine,
    combine_ons,
    inner,
    is_orthogonal,
    multiple_combine,
    multiple_combine_ons,
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

    def test_for_success__is_pure_qubits(self, valid_pure_qubits_amp):
        """_is_pure_qubitsメソッドの正常系テスト"""

        array = np.array(valid_pure_qubits_amp)
        assert _is_pure_qubits(array)

    def test_for_failure__is_pure_qubits(self, invalid_pure_qubits_amp):
        """_is_pure_qubitsメソッドの異常系テスト"""

        array = np.array(invalid_pure_qubits_amp)
        assert not _is_pure_qubits(array)

    def test_for_success__count_qubits(self, dict_for_test__count_qubits):
        """_count_qubitsメソッドの正常系テスト"""

        array = np.array(dict_for_test__count_qubits["array"])
        assert _count_qubits(array) == dict_for_test__count_qubits["count"]

    def test_for_success__resolve_arrays(
        self, dict_for_test_pure_qubits__resolve_arrays
    ):
        """_resolve_arraysメソッドの正常系テスト"""

        target = np.array(dict_for_test_pure_qubits__resolve_arrays["target"])
        vector, ndarray = _resolve_arrays(target)
        expected_vector = np.array(dict_for_test_pure_qubits__resolve_arrays["vector"])
        expected_ndarray = np.array(
            dict_for_test_pure_qubits__resolve_arrays["ndarray"]
        )
        assert np.all(vector == expected_vector)
        assert vector.shape == expected_vector.shape
        assert np.all(ndarray == expected_ndarray)
        assert ndarray.shape == expected_ndarray.shape

    def test_for_success_constructor(self, dict_for_test_pure_qubits_constructor):
        """
        __init__メソッドの正常系テスト
        """
        qubits = PureQubits(dict_for_test_pure_qubits_constructor["target"])
        assert allclose(
            qubits.projection, dict_for_test_pure_qubits_constructor["projection"]
        )
        assert allclose(
            qubits.projection_matrix,
            dict_for_test_pure_qubits_constructor["projection_matrix"],
        )
        assert (
            qubits.projection_matrix_dim
            == dict_for_test_pure_qubits_constructor["projection_matrix_dim"]
        )
        with captured_stdout() as stdout:
            qubits.dirac_notation()
        assert (
            stdout.getvalue() == dict_for_test_pure_qubits_constructor["dirac_notation"]
        )

    def test_for_failure_constructor(self, invalid_pure_qubits_amp):
        """
        __init__メソッドの異常系テスト
        """
        with pytest.raises(InitializeError):
            PureQubits(invalid_pure_qubits_amp)

    def test_for_success_combine(self, dict_for_test_pure_qubits_combine):
        """
        combineメソッドの正常系テスト
        """
        target_0 = PureQubits(dict_for_test_pure_qubits_combine["target_0"])
        target_1 = PureQubits(dict_for_test_pure_qubits_combine["target_1"])
        result = combine(target_0, target_1)

        expected_result = PureQubits(dict_for_test_pure_qubits_combine["result"])
        assert allclose(result.ndarray, expected_result.ndarray)
        assert allclose(result.vector, expected_result.vector)
        assert result.qubit_count == expected_result.qubit_count
        assert allclose(result.projection, expected_result.projection)
        assert allclose(result.projection_matrix, expected_result.projection_matrix)
        assert result.projection_matrix_dim == expected_result.projection_matrix_dim

    def test_for_success_multiple_combine(
        self, dict_for_test_pure_qubits_multiple_combine
    ):
        """
        multiple_combineメソッドの正常系テスト
        """
        target_list = dict_for_test_pure_qubits_multiple_combine["target_list"]
        result = multiple_combine(target_list)

        expected_result = PureQubits(
            dict_for_test_pure_qubits_multiple_combine["result"]
        )
        assert allclose(result.ndarray, expected_result.ndarray)
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


class TestOrthogonalSystem:
    """
    OrthogonalSystemクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_onb_constructor(self, dict_for_test_success_onb_constructor):
        """
        __init__メソッドの正常系テスト
        """
        qubits_list = dict_for_test_success_onb_constructor
        onb = OrthogonalSystem(qubits_list)
        assert onb.qubits_list == qubits_list

    def test_for_non_orthogonal_onb_constructor(
        self, dict_for_test_non_orthogonal_onb_constructor
    ):
        """
        __init__メソッドの異常系テスト
        (非直交)
        """
        with pytest.raises(InitializeError) as error:
            qubits_list = dict_for_test_non_orthogonal_onb_constructor
            OrthogonalSystem(qubits_list)
        assert "与えられたQubit群のリストは互いに直交しません" in str(error.value)

    def test_for_success_is_onb(self, dict_for_test_success_onb):
        """
        onbメソッドの正常系テスト (True)
        """
        qubits_list = dict_for_test_success_onb
        ons = OrthogonalSystem(qubits_list)
        assert ons.is_onb()

    def test_for_failure_is_onb(self, dict_for_test_failure_onb):
        """
        onbメソッドの正常系テスト (False)
        """
        qubits_list = dict_for_test_failure_onb
        ons = OrthogonalSystem(qubits_list)
        assert not ons.is_onb()

    def test_for_combine_ons(self, dict_for_test_combine_ons):
        """combin_onsメソッドの異常系テスト"""
        ons_0 = OrthogonalSystem(dict_for_test_combine_ons["ons_0"])
        ons_1 = OrthogonalSystem(dict_for_test_combine_ons["ons_1"])
        expected_result = OrthogonalSystem(dict_for_test_combine_ons["result"])

        result = combine_ons(ons_0, ons_1)

        assert len(result.qubits_list) == len(expected_result.qubits_list)
        for index in range(len(result.qubits_list)):
            qubits_0 = result.qubits_list[index]
            qubits_1 = expected_result.qubits_list[index]
            allclose(qubits_0.vector, qubits_1.vector)

    def test_for_multiple_combine_ons(self, dict_for_test_multiple_combine_ons):
        """multiple_combin_onsメソッドの異常系テスト"""
        ons_list = dict_for_test_multiple_combine_ons["ons_list"]
        expected_result = OrthogonalSystem(dict_for_test_multiple_combine_ons["result"])

        result = multiple_combine_ons(ons_list)

        assert len(result.qubits_list) == len(expected_result.qubits_list)
        for index in range(len(result.qubits_list)):
            qubits_0 = result.qubits_list[index]
            qubits_1 = expected_result.qubits_list[index]
            allclose(qubits_0.vector, qubits_1.vector)
