import random

import numpy as np
import pytest

from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.observable import (
    Observable,
    _resolve_observed_results,
    combine,
    create_from_ons,
    multiple_combine,
    observe,
)
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.base.utils import allclose, isclose


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

        eigen_values = observable.eigen_values
        eigen_states = observable.eigen_states
        matrix = observable.matrix
        matrix_dim = observable.matrix_dim
        ndarray = observable.ndarray

        expected_eigen_values = test_for_success_observable_constructor["eigen_values"]
        expected_eigen_states = test_for_success_observable_constructor["eigen_states"]
        expected_matrix = np.array(test_for_success_observable_constructor["matrix"])
        expected_matrix_dim = test_for_success_observable_constructor["matrix_dim"]
        expected_ndarray = np.array(test_for_success_observable_constructor["ndarray"])

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
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_for_success_create_from_ons(self, test_for_success_create_from_ons):
        """
        create_from_onsメソッドの正常系テスト
        """
        observed_values = test_for_success_create_from_ons["observed_values"]
        ons = test_for_success_create_from_ons["ons"]
        observable = create_from_ons(observed_values, ons)

        eigen_values = observable.eigen_values
        eigen_states = observable.eigen_states
        matrix = observable.matrix
        matrix_dim = observable.matrix_dim
        ndarray = observable.ndarray

        expected_eigen_values = test_for_success_create_from_ons["eigen_values"]
        expected_eigen_states = test_for_success_create_from_ons["eigen_states"]
        expected_matrix = np.array(test_for_success_create_from_ons["matrix"])
        expected_matrix_dim = test_for_success_create_from_ons["matrix_dim"]
        expected_ndarray = np.array(test_for_success_create_from_ons["ndarray"])

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
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

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
        assert isclose(expected_value, expected_expected_value)

    def test_for_success__resolve_observed_results(
        self, test_for_success__resolve_observed_results
    ):
        """
        _resolve_observed_resultsメソッドの正常系テスト
        """
        eigen_values = test_for_success__resolve_observed_results["eigen_values"]
        eigen_states = test_for_success__resolve_observed_results["eigen_states"]

        unique_eigen_values, projections = _resolve_observed_results(
            eigen_values, eigen_states
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
                if isclose(
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

        observed_value = observe(observable, target)

        expected_observed_value = test_for_success_observe["expected_observed_value"]
        expected_qubits = test_for_success_observe["expected_qubits"]

        assert isclose(observed_value, expected_observed_value)
        assert allclose(target.matrix, expected_qubits.matrix)

    def test_for_success_observable_combine(self, test_for_success_observable_combine):
        """
        combineメソッドの正常系テスト
        """
        target_0 = test_for_success_observable_combine["target_0"]
        target_1 = test_for_success_observable_combine["target_1"]
        observable = combine(target_0, target_1)

        eigen_values = observable.eigen_values
        eigen_states = observable.eigen_states
        matrix = observable.matrix
        matrix_dim = observable.matrix_dim
        ndarray = observable.ndarray

        expected_eigen_values = test_for_success_observable_combine["eigen_values"]
        expected_eigen_states = test_for_success_observable_combine["eigen_states"]
        expected_matrix = np.array(test_for_success_observable_combine["matrix"])
        expected_matrix_dim = test_for_success_observable_combine["matrix_dim"]
        expected_ndarray = np.array(test_for_success_observable_combine["ndarray"])

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
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_for_success_observable_multiple_combine(
        self, test_for_success_observable_multiple_combine
    ):
        """
        multiple_combineメソッドの正常系テスト
        """
        target_list = test_for_success_observable_multiple_combine["target_list"]
        observable = multiple_combine(target_list)

        eigen_values = observable.eigen_values
        eigen_states = observable.eigen_states
        matrix = observable.matrix
        matrix_dim = observable.matrix_dim
        ndarray = observable.ndarray

        expected_eigen_values = test_for_success_observable_multiple_combine[
            "eigen_values"
        ]
        expected_eigen_states = test_for_success_observable_multiple_combine[
            "eigen_states"
        ]
        expected_matrix = np.array(
            test_for_success_observable_multiple_combine["matrix"]
        )
        expected_matrix_dim = test_for_success_observable_multiple_combine["matrix_dim"]
        expected_ndarray = np.array(
            test_for_success_observable_multiple_combine["ndarray"]
        )

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
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    # def test_valid_observable_for_one_qubit(
    #     self, valid_observed_values, observed_basis
    # ):
    #     """[正常系]: 単一Qubit系に対する妥当な観測量"""
    #     observable = Observable(valid_observed_values, observed_basis)
    #     for index in range(len(valid_observed_values)):
    #         assert observable.elements[index]["value"] == valid_observed_values[index]
    #         assert np.all(
    #             observable.elements[index]["qubits"].array
    #             == observed_basis.qubits_group[index].array
    #         )

    # def test_invalid_observable_for_one_qubit(
    #     self, invalid_observed_values, observed_basis
    # ):
    #     """[異常系]: 単一Qubit系に対する妥当でない観測量を構成時、エラーとなること"""
    #     with pytest.raises(InitializeError):
    #         Observable(invalid_observed_values, observed_basis)

    # def test_valid_observable_for_multiple_qubit(
    #     self, valid_multi_particles_observed_values, multi_particles_observed_basis
    # ):
    #     """[正常系]: 単一Qubit系に対する妥当な観測量"""
    #     observable = Observable(
    #         valid_multi_particles_observed_values, multi_particles_observed_basis
    #     )
    #     for index in range(len(valid_multi_particles_observed_values)):
    #         assert (
    #             observable.elements[index]["value"]
    #             == valid_multi_particles_observed_values[index]
    #         )
    #         assert np.all(
    #             observable.elements[index]["qubits"].array
    #             == multi_particles_observed_basis.qubits_group[index].array
    #         )

    # def test_invalid_observable_for_multiple_qubit(
    #     self, invalid_multi_particles_observed_values, multi_particles_observed_basis
    # ):
    #     """[異常系]: 単一Qubit系に対する妥当でない観測量を構成時、エラーとなること"""
    #     with pytest.raises(InitializeError):
    #         Observable(
    #             invalid_multi_particles_observed_values, multi_particles_observed_basis
    #         )

    # def test_expected_value_for_observable_with_one_qubit(
    #     self, dict_for_test_expected_value
    # ):
    #     """[正常系]: 単一Qubitに対する観測量の期待値"""
    #     dict_for_test = dict_for_test_expected_value
    #     expected_value = dict_for_test["observable"].expected_value(
    #         dict_for_test["qubit"]
    #     )
    #     assert round(expected_value, APPROX_DIGIT) == dict_for_test["expected_value"]

    # def test_observation_for_one_qubit(self, dict_for_test_observation):
    #     """[正常系]: von Neumann観測による単一Qubitに対する観測"""
    #     dict_for_test = dict_for_test_observation

    #     # 観測量の第一成分の値と遷移後状態を期待する
    #     expected_result = dict_for_test["observable"].elements[0]

    #     # seedを固定してテストを可能にする
    #     random.seed(dict_for_test["randomize_seed"])

    #     # 観測実施
    #     observed_value = dict_for_test["observable"].observe(dict_for_test["qubit"])

    #     assert observed_value == expected_result["value"]
    #     assert np.all(
    #         np.round(dict_for_test["qubit"].array - expected_result["qubits"].array)
    #         == 0.0
    #     )

    # def test_expected_value_for_observable_with_multiple_qubit(
    #     self, dict_for_test_expected_value_with_compound_observable
    # ):
    #     """[正常系]: 3粒子Qubit系に対する観測量の期待値"""
    #     dict_for_test = dict_for_test_expected_value_with_compound_observable
    #     expected_value = dict_for_test["observable"].expected_value(
    #         dict_for_test["qubits"]
    #     )
    #     assert round(expected_value, APPROX_DIGIT) == dict_for_test["expected_value"]

    # def test_observation_for_multiple_qubit(
    #     self, dict_for_test_observation_with_compound_observable
    # ):
    #     """[正常系]: von Neumann観測による3粒子Qubit系に対する観測"""
    #     dict_for_test = dict_for_test_observation_with_compound_observable

    #     # 観測量の第一成分の値と遷移後状態を期待する
    #     expected_result = dict_for_test["observable"].elements[0]

    #     # seedを固定してテストを可能にする
    #     random.seed(dict_for_test["randomize_seed"])

    #     # 観測実施
    #     observed_value = dict_for_test["observable"].observe(dict_for_test["qubits"])

    #     assert observed_value == expected_result["value"]
    #     assert np.all(
    #         np.round(dict_for_test["qubits"].array - expected_result["qubits"].array)
    #         == 0.0
    #     )

    # def test_combine_observables(self, dict_for_test_combine_observables):
    #     """[正常系]: 2粒子Qubit系に対する観測量同士の結合"""
    #     dict_for_test = dict_for_test_combine_observables
    #     combined_observable = combine(
    #         dict_for_test["observable_group"][0], dict_for_test["observable_group"][1]
    #     )

    #     assert np.all(
    #         np.round(
    #             combined_observable.matrix - np.array(dict_for_test["expected_matrix"])
    #         )
    #         == 0.0
    #     )
