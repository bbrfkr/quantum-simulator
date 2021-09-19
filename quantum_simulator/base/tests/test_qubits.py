import numpy
import pytest

from ...base.utils import allclose
from ..error import InvalidProbabilitiesError, NotMatchCountError, ReductionError
from ..qubits import (
    Qubits,
    combine,
    convex_combination,
    create_from_ons,
    generalize,
    is_qubits_dim,
    multiple_combine,
    multiple_reduction,
    reduction,
    specialize,
)


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

        array = numpy.array(valid_qubits_array)
        assert is_qubits_dim(array)

    def test_for_failure_is_qubits_dim(self, invalid_qubits_array):
        """is_qubits_dimメソッドの異常系テスト"""

        array = numpy.array(invalid_qubits_array)
        assert not is_qubits_dim(array)

    def test_for_success_constructor(self, dict_for_test_qubits_constructor):
        """__init__メソッドの正常系テスト"""
        target = dict_for_test_qubits_constructor["target"]
        qubits = Qubits(target)
        matrix = qubits.matrix
        qubit_count = qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_qubits_constructor["matrix"])
        expected_qubit_count = dict_for_test_qubits_constructor["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_generalize(self, dict_for_test_generalize):
        """generalizeメソッドの正常系テスト"""
        target = dict_for_test_generalize["target"]
        qubits = generalize(target)

        matrix = qubits.matrix
        qubit_count = qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_generalize["matrix"])
        expected_qubit_count = dict_for_test_generalize["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_specialize(self, dict_for_test_specialize):
        """specializeメソッドの正常系テスト"""
        target = dict_for_test_specialize["target"]
        result = specialize(target)

        expected_vector = dict_for_test_specialize["vector"]
        expected_qubit_count = dict_for_test_specialize["qubit_count"]
        assert allclose(result.vector, expected_vector)
        assert result.qubit_count == expected_qubit_count

    def test_for_success_convex_combination(self, dict_for_test_convex_combination):
        """convex_combinationメソッドの正常系テスト"""
        probabilities = dict_for_test_convex_combination["probabilities"]
        qubits_list = dict_for_test_convex_combination["qubits_list"]
        qubits = convex_combination(probabilities, qubits_list)

        matrix = qubits.matrix
        qubit_count = qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_convex_combination["matrix"])
        expected_qubit_count = dict_for_test_convex_combination["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_create_from_ons(self, dict_for_test_create_from_ons):
        """create_from_onsメソッドの正常系テスト"""
        probabilities = dict_for_test_create_from_ons["probabilities"]
        ons = dict_for_test_create_from_ons["ons"]
        qubits = create_from_ons(probabilities, ons)

        matrix = qubits.matrix
        qubit_count = qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_create_from_ons["matrix"])
        expected_qubit_count = dict_for_test_create_from_ons["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

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

        matrix = qubits.matrix
        qubit_count = qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_qubits_combine["matrix"])
        expected_qubit_count = dict_for_test_qubits_combine["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_reduction(self, dict_for_test_reduction):
        """reductionメソッドの正常系テスト"""
        qubits = dict_for_test_reduction["qubits"]
        target_particle = dict_for_test_reduction["target_particle"]
        reduced_qubits = reduction(qubits, target_particle)

        matrix = reduced_qubits.matrix
        qubit_count = reduced_qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_reduction["matrix"])
        expected_qubit_count = dict_for_test_reduction["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_multiple_combine(self, dict_for_test_qubits_multiple_combine):
        """multiple_combineメソッドの正常系テスト"""
        qubits_list = dict_for_test_qubits_multiple_combine["qubits_list"]
        qubits = multiple_combine(qubits_list)

        matrix = qubits.matrix
        qubit_count = qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_qubits_multiple_combine["matrix"])
        expected_qubit_count = dict_for_test_qubits_multiple_combine["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_multiple_reduction(self, dict_for_test_multiple_reduction):
        """multiple_reductionメソッドの正常系テスト"""
        qubits = dict_for_test_multiple_reduction["qubits"]
        target_particles = dict_for_test_multiple_reduction["target_particles"]
        reduced_qubits = multiple_reduction(qubits, target_particles)

        matrix = reduced_qubits.matrix
        qubit_count = reduced_qubits.qubit_count

        expected_matrix = numpy.array(dict_for_test_multiple_reduction["matrix"])
        expected_qubit_count = dict_for_test_multiple_reduction["qubit_count"]

        assert qubit_count == expected_qubit_count
        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape
