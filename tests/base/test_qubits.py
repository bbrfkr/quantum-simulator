import numpy
import pytest

from quantum_simulator.base.error import (
    InvalidProbabilitiesError,
    NotMatchCountError,
    ReductionError,
)
from quantum_simulator.base.qubits import (
    Qubits,
    combine,
    convex_combination,
    create_from_ons,
    generalize,
    is_qubits_dim,
    multiple_combine,
    multiple_reduction,
    reduction,
    resolve_arrays,
    resolve_eigen,
    specialize,
)
from quantum_simulator.base.switch_cupy import xp_factory
from quantum_simulator.base.utils import allclose

np = xp_factory()  # typing: numpy


class TestQubits:
    """
    Qubitsクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_is_qubits_dim(self, valid_qubits_array):
        """is_qubits_dimメソッドの正常系テスト"""

        array = np.array(valid_qubits_array)
        assert is_qubits_dim(array)

    def test_for_failure_is_qubits_dim(self, invalid_qubits_array):
        """is_qubits_dimメソッドの異常系テスト"""

        array = np.array(invalid_qubits_array)
        assert not is_qubits_dim(array)

    def test_for_success_resolve_arrays(self, dict_for_test_qubits_resolve_arrays):
        """resolve_arraysメソッドの正常系テスト"""
        target = np.array(dict_for_test_qubits_resolve_arrays["target"])
        matrix, ndarray = resolve_arrays(target)
        expected_matrix = np.array(dict_for_test_qubits_resolve_arrays["matrix"])
        expected_ndarray = np.array(dict_for_test_qubits_resolve_arrays["ndarray"])
        assert numpy.allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape
        assert ndarray.shape == expected_ndarray.shape

    def test_for_success_resolve_eigen(self, dict_for_test_resolve_eigen):
        """resolve_eigenメソッドの正常系テスト"""
        target = np.array(dict_for_test_resolve_eigen["target"])
        eigen_values, eigen_states = resolve_eigen(target)

        expected_eigen_values = dict_for_test_resolve_eigen["eigen_values"]
        expected_eigen_states = dict_for_test_resolve_eigen["eigen_states"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
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
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_qubits_constructor["eigen_values"]
        expected_eigen_states = dict_for_test_qubits_constructor["eigen_states"]
        expected_matrix = np.array(dict_for_test_qubits_constructor["matrix"])
        expected_ndarray = np.array(dict_for_test_qubits_constructor["ndarray"])
        expected_qubit_count = dict_for_test_qubits_constructor["qubit_count"]
        expected_is_pure = dict_for_test_qubits_constructor["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_generalize(self, dict_for_test_generalize):
        """generalizeメソッドの正常系テスト"""
        target = dict_for_test_generalize["target"]
        qubits = generalize(target)

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_generalize["eigen_values"]
        expected_eigen_states = dict_for_test_generalize["eigen_states"]
        expected_matrix = np.array(dict_for_test_generalize["matrix"])
        expected_ndarray = np.array(dict_for_test_generalize["ndarray"])
        expected_qubit_count = dict_for_test_generalize["qubit_count"]
        expected_is_pure = dict_for_test_generalize["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_specialize(self, dict_for_test_specialize):
        """specializeメソッドの正常系テスト"""
        target = dict_for_test_specialize["target"]
        result = specialize(target)

        expected_ndarray = dict_for_test_specialize["ndarray"]
        expected_vector = dict_for_test_specialize["vector"]
        expected_qubit_count = dict_for_test_specialize["qubit_count"]
        assert allclose(result.ndarray, expected_ndarray)
        assert allclose(result.vector, expected_vector)
        assert result.qubit_count == expected_qubit_count

    def test_for_success_convex_combination(self, dict_for_test_convex_combination):
        """convex_combinationメソッドの正常系テスト"""
        probabilities = dict_for_test_convex_combination["probabilities"]
        qubits_list = dict_for_test_convex_combination["qubits_list"]
        qubits = convex_combination(probabilities, qubits_list)

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_convex_combination["eigen_values"]
        expected_eigen_states = dict_for_test_convex_combination["eigen_states"]
        expected_matrix = np.array(dict_for_test_convex_combination["matrix"])
        expected_ndarray = np.array(dict_for_test_convex_combination["ndarray"])
        expected_qubit_count = dict_for_test_convex_combination["qubit_count"]
        expected_is_pure = dict_for_test_convex_combination["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_create_from_ons(self, dict_for_test_create_from_ons):
        """create_from_onsメソッドの正常系テスト"""
        probabilities = dict_for_test_create_from_ons["probabilities"]
        ons = dict_for_test_create_from_ons["ons"]
        qubits = create_from_ons(probabilities, ons)

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_create_from_ons["eigen_values"]
        expected_eigen_states = dict_for_test_create_from_ons["eigen_states"]
        expected_matrix = np.array(dict_for_test_create_from_ons["matrix"])
        expected_ndarray = np.array(dict_for_test_create_from_ons["ndarray"])
        expected_qubit_count = dict_for_test_create_from_ons["qubit_count"]
        expected_is_pure = dict_for_test_create_from_ons["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_invalid_probabilities_error_convex_combination(
        self, invalid_probabilities_and_qubits_list
    ):
        """convex_combinationメソッドに対する不正な確率リストエラーの異常系テスト"""
        with pytest.raises(InvalidProbabilitiesError):
            probabilities = invalid_probabilities_and_qubits_list["probabilities"]
            qubits_list = invalid_probabilities_and_qubits_list["qubits_list"]
            convex_combination(probabilities, qubits_list)

    def test_for_not_match_count_error_convex_combination(
        self, not_match_count_probabilities_and_qubits_list
    ):
        """convex_combinationメソッドに対するリスト要素数不一致エラーの異常系テスト"""
        with pytest.raises(NotMatchCountError):
            probabilities = not_match_count_probabilities_and_qubits_list[
                "probabilities"
            ]
            qubits_list = not_match_count_probabilities_and_qubits_list["qubits_list"]
            convex_combination(probabilities, qubits_list)

    def test_for_failure_reduction(self, invalid_reduction):
        """reductionメソッドの正常系テスト"""
        qubits = invalid_reduction["qubits"]
        target_particle = invalid_reduction["target_particle"]

        with pytest.raises(ReductionError):
            reduction(qubits, target_particle)

    def test_for_success_combine(self, dict_for_test_qubits_combine):
        """combineメソッドの正常系テスト"""
        qubits_list = dict_for_test_qubits_combine["qubits_list"]
        qubits = combine(qubits_list[0], qubits_list[1])

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_qubits_combine["eigen_values"]
        expected_eigen_states = dict_for_test_qubits_combine["eigen_states"]
        expected_matrix = np.array(dict_for_test_qubits_combine["matrix"])
        expected_ndarray = np.array(dict_for_test_qubits_combine["ndarray"])
        expected_qubit_count = dict_for_test_qubits_combine["qubit_count"]
        expected_is_pure = dict_for_test_qubits_combine["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

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
        ndarray = reduced_qubits.ndarray
        qubit_count = reduced_qubits.qubit_count
        is_pure = reduced_qubits.is_pure()

        expected_eigen_values = dict_for_test_reduction["eigen_values"]
        expected_eigen_states = dict_for_test_reduction["eigen_states"]
        expected_matrix = np.array(dict_for_test_reduction["matrix"])
        expected_ndarray = np.array(dict_for_test_reduction["ndarray"])
        expected_qubit_count = dict_for_test_reduction["qubit_count"]
        expected_is_pure = dict_for_test_reduction["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_multiple_combine(self, dict_for_test_qubits_multiple_combine):
        """multiple_combineメソッドの正常系テスト"""
        qubits_list = dict_for_test_qubits_multiple_combine["qubits_list"]
        qubits = multiple_combine(qubits_list)

        eigen_values = qubits.eigen_values
        eigen_states = qubits.eigen_states
        matrix = qubits.matrix
        ndarray = qubits.ndarray
        qubit_count = qubits.qubit_count
        is_pure = qubits.is_pure()

        expected_eigen_values = dict_for_test_qubits_multiple_combine["eigen_values"]
        expected_eigen_states = dict_for_test_qubits_multiple_combine["eigen_states"]
        expected_matrix = np.array(dict_for_test_qubits_multiple_combine["matrix"])
        expected_ndarray = np.array(dict_for_test_qubits_multiple_combine["ndarray"])
        expected_qubit_count = dict_for_test_qubits_multiple_combine["qubit_count"]
        expected_is_pure = dict_for_test_qubits_multiple_combine["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure

    def test_for_success_multiple_reduction(self, dict_for_test_multiple_reduction):
        """multiple_reductionメソッドの正常系テスト"""
        qubits = dict_for_test_multiple_reduction["qubits"]
        target_particles = dict_for_test_multiple_reduction["target_particles"]
        reduced_qubits = multiple_reduction(qubits, target_particles)

        eigen_values = reduced_qubits.eigen_values
        eigen_states = reduced_qubits.eigen_states
        matrix = reduced_qubits.matrix
        ndarray = reduced_qubits.ndarray
        qubit_count = reduced_qubits.qubit_count
        is_pure = reduced_qubits.is_pure()

        expected_eigen_values = dict_for_test_multiple_reduction["eigen_values"]
        expected_eigen_states = dict_for_test_multiple_reduction["eigen_states"]
        expected_matrix = np.array(dict_for_test_multiple_reduction["matrix"])
        expected_ndarray = np.array(dict_for_test_multiple_reduction["ndarray"])
        expected_qubit_count = dict_for_test_multiple_reduction["qubit_count"]
        expected_is_pure = dict_for_test_multiple_reduction["is_pure"]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index], eigen_values[result_index]
                ) and allclose(
                    expected_eigen_states[expected_index],
                    eigen_states[result_index].vector,
                ):
                    is_passed = True

            assert is_passed

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)
        assert is_pure == expected_is_pure
