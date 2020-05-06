import numpy as np
import pytest

from quantum_simulator.base.error import (
    InitializeError,
    InvalidProbabilitiesError,
    NotMatchCountError,
    ReductionError,
)
from quantum_simulator.base.qubits import (
    Qubits,
    _is_qubits_dim,
    _resolve_arrays,
    _resolve_eigen,
    combine,
    create_from_qubits_list,
    reduction,
)
from quantum_simulator.base.utils import allclose, isclose


class TestQubits:
    """
    Qubitsクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success__is_qubits_dim(self, valid_qubits_array):
        """_is_qubits_dimメソッドの正常系テスト"""

        array = np.array(valid_qubits_array)
        assert _is_qubits_dim(array)

    def test_for_failure__is_qubits_dim(self, invalid_qubits_array):
        """_is_qubits_dimメソッドの異常系テスト"""

        array = np.array(invalid_qubits_array)
        assert not _is_qubits_dim(array)

    def test_for_success__resolve_arrays(self, dict_for_test_qubits__resolve_arrays):
        """_resolve_arraysメソッドの正常系テスト"""
        target = np.array(dict_for_test_qubits__resolve_arrays["target"])
        matrix, ndarray = _resolve_arrays(target)
        expected_matrix = np.array(dict_for_test_qubits__resolve_arrays["matrix"])
        expected_ndarray = np.array(dict_for_test_qubits__resolve_arrays["ndarray"])
        assert np.allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape
        assert ndarray.shape == expected_ndarray.shape

    def test_for_success__resolve_eigen(self, dict_for_test__resolve_eigen):
        """_resolve_eigenメソッドの正常系テスト"""
        target = np.array(dict_for_test__resolve_eigen["target"])
        eigen_values, eigen_states = _resolve_eigen(target)

        expected_eigen_values = dict_for_test__resolve_eigen["eigen_values"]
        expected_eigen_states = dict_for_test__resolve_eigen["eigen_states"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if isclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

    def test_for_success_constructor(self, dict_for_test_qubits_constructor):
        """__init__メソッドの正常系テスト"""
        target = dict_for_test_qubits_constructor["target"]
        qubits = Qubits(target)
        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        matrix_dim = qubits.matrix_dim
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_qubits_constructor["eigen_values"]
        expected_eigen_states = dict_for_test_qubits_constructor["eigen_states"]
        expected_matrix = np.array(dict_for_test_qubits_constructor["matrix"])
        expected_matrix_dim = dict_for_test_qubits_constructor["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_qubits_constructor["ndarray"])
        expected_qubit_count = dict_for_test_qubits_constructor["qubit_count"]
        expected_is_pure = dict_for_test_qubits_constructor["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if isclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert matrix_dim == expected_matrix_dim
        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_constructor(self):
        """__init__メソッド 虚数固有値の異常系テスト"""
        with pytest.raises(InitializeError) as error:
            target = [[0.5 + 0j, 0j], [0j, 0.5 * 1j]]
            Qubits(target)
        assert "与えられたリストには虚数の固有値が存在します" in str(error.value)

    def test_for_success_create_from_qubits_list(
        self, dict_for_test_create_from_qubits_list
    ):
        """create_from_qubits_listメソッドの正常系テスト"""
        probabilities = dict_for_test_create_from_qubits_list["probabilities"]
        qubits_list = dict_for_test_create_from_qubits_list["qubits_list"]
        qubits = create_from_qubits_list(probabilities, qubits_list)

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        matrix_dim = qubits.matrix_dim
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_create_from_qubits_list["eigen_values"]
        expected_eigen_states = dict_for_test_create_from_qubits_list["eigen_states"]
        expected_matrix = np.array(dict_for_test_create_from_qubits_list["matrix"])
        expected_matrix_dim = dict_for_test_create_from_qubits_list["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_create_from_qubits_list["ndarray"])
        expected_qubit_count = dict_for_test_create_from_qubits_list["qubit_count"]
        expected_is_pure = dict_for_test_create_from_qubits_list["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if isclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert matrix_dim == expected_matrix_dim
        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_invalid_probabilities_error_create_from_qubits_list(
        self, invalid_probabilities_and_qubits_list
    ):
        """create_from_qubits_listメソッドに対する不正な確率リストエラーの異常系テスト"""
        with pytest.raises(InvalidProbabilitiesError):
            probabilities = invalid_probabilities_and_qubits_list["probabilities"]
            qubits_list = invalid_probabilities_and_qubits_list["qubits_list"]
            create_from_qubits_list(probabilities, qubits_list)

    def test_for_not_match_count_error_create_from_qubits_list(
        self, not_match_count_probabilities_and_qubits_list
    ):
        """create_from_qubits_listメソッドに対するリスト要素数不一致エラーの異常系テスト"""
        with pytest.raises(NotMatchCountError):
            probabilities = not_match_count_probabilities_and_qubits_list[
                "probabilities"
            ]
            qubits_list = not_match_count_probabilities_and_qubits_list["qubits_list"]
            create_from_qubits_list(probabilities, qubits_list)

    def test_for_success_combine(self, dict_for_test_qubits_combine):
        """combineメソッドの正常系テスト"""
        qubits_list = dict_for_test_qubits_combine["qubits_list"]
        qubits = combine(qubits_list[0], qubits_list[1])

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        matrix_dim = qubits.matrix_dim
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_qubits_combine["eigen_values"]
        expected_eigen_states = dict_for_test_qubits_combine["eigen_states"]
        expected_matrix = np.array(dict_for_test_qubits_combine["matrix"])
        expected_matrix_dim = dict_for_test_qubits_combine["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_qubits_combine["ndarray"])
        expected_qubit_count = dict_for_test_qubits_combine["qubit_count"]
        expected_is_pure = dict_for_test_qubits_combine["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if isclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert matrix_dim == expected_matrix_dim
        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_reduction(self, dict_for_test_reduction):
        """reductionメソッドの正常系テスト"""
        qubits = dict_for_test_reduction["qubits"]
        target_particle = dict_for_test_reduction["target_particle"]
        reduced_qubits = reduction(qubits, target_particle)

        eigen_values = reduced_qubits.eigen_values
        eigen_states = reduced_qubits.eigen_states
        matrix = reduced_qubits.matrix
        matrix_dim = reduced_qubits.matrix_dim
        ndarray = reduced_qubits.ndarray
        qubit_count = reduced_qubits.qubit_count
        is_pure = reduced_qubits.is_pure()

        expected_eigen_values = dict_for_test_reduction["eigen_values"]
        expected_eigen_states = dict_for_test_reduction["eigen_states"]
        expected_matrix = np.array(dict_for_test_reduction["matrix"])
        expected_matrix_dim = dict_for_test_reduction["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_reduction["ndarray"])
        expected_qubit_count = dict_for_test_reduction["qubit_count"]
        expected_is_pure = dict_for_test_reduction["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if isclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert matrix_dim == expected_matrix_dim
        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_failure_reduction(self, invalid_reduction):
        """reductionメソッドの正常系テスト"""
        qubits = invalid_reduction["qubits"]
        target_particle = invalid_reduction["target_particle"]

        with pytest.raises(ReductionError):
            reduction(qubits, target_particle)
