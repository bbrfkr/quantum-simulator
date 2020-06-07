import random

import numpy

from quantum_simulator.base.observable import (
    Observable,
    _resolve_observed_results,
    combine,
    create_from_ons,
    multiple_combine,
    observe,
)
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.base.utils import allclose


class TestObservable:
    """
    Observableクラスと付随するメソッドのテスト
        * 以下のロジックはテストしない
            * ただ値を代入するだけのロジック
            * すでにテスト済みの値を代入するロジック
            * 既存オブジェクトを出力するだけのロジック
    """

    def test_for_success_constructor(self, test_for_success_observable_constructor):
        """
        __init__メソッドの正常系テスト
        """
        target = test_for_success_observable_constructor["target"]
        observable = Observable(target)

        matrix = observable.matrix

        expected_matrix = numpy.array(test_for_success_observable_constructor["matrix"])

        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_create_from_ons(self, test_for_success_create_from_ons):
        """
        create_from_onsメソッドの正常系テスト
        """
        observed_values = test_for_success_create_from_ons["observed_values"]
        ons = test_for_success_create_from_ons["ons"]
        observable = create_from_ons(observed_values, ons)

        matrix = observable.matrix

        expected_matrix = numpy.array(test_for_success_create_from_ons["matrix"])

        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_expected_value(self, test_for_success_expected_value_for_pure):
        """
        expected_valueメソッドの純粋状態に対する正常系テスト
        """
        observable = Observable(test_for_success_expected_value_for_pure["observable"])
        target = Qubits(test_for_success_expected_value_for_pure["target"])
        expected_value = observable.expected_value(target)

        expected_expected_value = test_for_success_expected_value_for_pure[
            "expected_value"
        ]
        assert allclose(expected_value, expected_expected_value)

    def test_for_success__resolve_observed_results(
        self, test_for_success__resolve_observed_results
    ):
        """
        _resolve_observed_resultsメソッドの正常系テスト
        """
        eigen_values = test_for_success__resolve_observed_results["eigen_values"]
        eigen_states = test_for_success__resolve_observed_results["eigen_states"]

        unique_eigen_values, projections = _resolve_observed_results(
            eigen_values, numpy.array(eigen_states)
        )

        expected_eigen_values = test_for_success__resolve_observed_results[
            "expected_eigen_values"
        ]
        expected_projections = test_for_success__resolve_observed_results[
            "expected_projections"
        ]

        for expected_index in range(len(expected_eigen_values)):
            is_passed = False

            for result_index in range(len(unique_eigen_values)):
                if allclose(
                    expected_eigen_values[expected_index],
                    unique_eigen_values[result_index],
                ) and allclose(
                    expected_projections[expected_index].matrix,
                    projections[result_index].matrix,
                ):
                    is_passed = True

            assert is_passed

    def test_for_success_observe(self, test_for_success_observe):
        """
        observeメソッドの正常系テスト
        """
        observable = test_for_success_observe["observable"]
        target = test_for_success_observe["target"]

        random.seed(test_for_success_observe["random_seed"])

        observed_value, converged_qubits = observe(observable, target)

        expected_observed_value = test_for_success_observe["expected_observed_value"]
        expected_qubits = test_for_success_observe["expected_qubits"]

        assert allclose(observed_value, expected_observed_value)
        assert allclose(converged_qubits.matrix, expected_qubits.matrix)

    def test_for_success_observable_combine(self, test_for_success_observable_combine):
        """
        combineメソッドの正常系テスト
        """
        target_0 = test_for_success_observable_combine["target_0"]
        target_1 = test_for_success_observable_combine["target_1"]
        observable = combine(target_0, target_1)

        matrix = observable.matrix

        expected_matrix = numpy.array(test_for_success_observable_combine["matrix"])

        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape

    def test_for_success_observable_multiple_combine(
        self, test_for_success_observable_multiple_combine
    ):
        """
        multiple_combineメソッドの正常系テスト
        """
        target_list = test_for_success_observable_multiple_combine["target_list"]
        observable = multiple_combine(target_list)

        matrix = observable.matrix

        expected_matrix = numpy.array(
            test_for_success_observable_multiple_combine["matrix"]
        )

        assert allclose(matrix, expected_matrix)
        assert matrix.shape == expected_matrix.shape
