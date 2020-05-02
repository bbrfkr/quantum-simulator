import random

import numpy as np
import pytest

from quantum_simulator.base.conf import APPROX_DIGIT
from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.observable import Observable, ObservedBasis, combine


class TestObservedBasis:
    """ObservedBasisクラスと付随するメソッドのテスト"""

    # 単一Qubitに対する観測基底のテスト
    def test_valid_observe_basis(self, orthogonal_qubits):
        """[正常系]: 直交した単一Qubit同士で構成する観測基底"""
        observe_basis = ObservedBasis(orthogonal_qubits)
        assert observe_basis.qubits_group == orthogonal_qubits

    def test_invalid_observe_basis(self, non_orthogonal_qubits):
        """[異常系]: 直交しない単一Qubit同士で観測基底を構成時、エラーとなること"""
        with pytest.raises(InitializeError) as error:
            ObservedBasis(non_orthogonal_qubits)
        assert "観測基底が直交しません" in str(error.value)

    # Qubit群に対する観測基底のテスト
    def test_valid_observe_basis_with_multiple_qubits(
        self, orthogonal_multiple_qubits_groups
    ):
        """[正常系]: 直交したQubit群同士で構成する観測基底"""
        observe_basis = ObservedBasis(orthogonal_multiple_qubits_groups)
        assert observe_basis.qubits_group == orthogonal_multiple_qubits_groups

    def test_invalid_observe_basis_with_non_orthogonal_multiple_qubits(
        self, non_orthogonal_multiple_qubits_groups
    ):
        """[異常系]: 直交しないQubit群同士で観測基底を構成時、エラーとなること"""
        with pytest.raises(InitializeError) as error:
            ObservedBasis(non_orthogonal_multiple_qubits_groups)
        assert "観測基底が直交しません" in str(error.value)


class TestObservable:
    """Observableクラスと付随するメソッドのテスト"""

    def test_valid_observable_for_one_qubit(
        self, valid_observed_values, observed_basis
    ):
        """[正常系]: 単一Qubit系に対する妥当な観測量"""
        observable = Observable(valid_observed_values, observed_basis)
        for index in range(len(valid_observed_values)):
            assert observable.elements[index]["value"] == valid_observed_values[index]
            assert np.all(
                observable.elements[index]["qubits"].amplitudes
                == observed_basis.qubits_group[index].amplitudes
            )

    def test_invalid_observable_for_one_qubit(
        self, invalid_observed_values, observed_basis
    ):
        """[異常系]: 単一Qubit系に対する妥当でない観測量を構成時、エラーとなること"""
        with pytest.raises(InitializeError):
            Observable(invalid_observed_values, observed_basis)

    def test_valid_observable_for_multiple_qubit(
        self, valid_multi_particles_observed_values, multi_particles_observed_basis
    ):
        """[正常系]: 単一Qubit系に対する妥当な観測量"""
        observable = Observable(
            valid_multi_particles_observed_values, multi_particles_observed_basis
        )
        for index in range(len(valid_multi_particles_observed_values)):
            assert (
                observable.elements[index]["value"]
                == valid_multi_particles_observed_values[index]
            )
            assert np.all(
                observable.elements[index]["qubits"].amplitudes
                == multi_particles_observed_basis.qubits_group[index].amplitudes
            )

    def test_invalid_observable_for_multiple_qubit(
        self, invalid_multi_particles_observed_values, multi_particles_observed_basis
    ):
        """[異常系]: 単一Qubit系に対する妥当でない観測量を構成時、エラーとなること"""
        with pytest.raises(InitializeError):
            Observable(
                invalid_multi_particles_observed_values, multi_particles_observed_basis
            )

    def test_expected_value_for_observable_with_one_qubit(
        self, dict_for_test_expected_value
    ):
        """[正常系]: 単一Qubitに対する観測量の期待値"""
        dict_for_test = dict_for_test_expected_value
        expected_value = dict_for_test["observable"].expected_value(
            dict_for_test["qubit"]
        )
        assert round(expected_value, APPROX_DIGIT) == dict_for_test["expected_value"]

    def test_observation_for_one_qubit(self, dict_for_test_observation):
        """[正常系]: von Neumann観測による単一Qubitに対する観測"""
        dict_for_test = dict_for_test_observation

        # 観測量の第一成分の値と遷移後状態を期待する
        expected_result = dict_for_test["observable"].elements[0]

        # seedを固定してテストを可能にする
        random.seed(dict_for_test["randomize_seed"])

        # 観測実施
        observed_value = dict_for_test["observable"].observe(dict_for_test["qubit"])

        assert observed_value == expected_result["value"]
        assert np.all(
            np.round(
                dict_for_test["qubit"].amplitudes - expected_result["qubits"].amplitudes
            )
            == 0.0
        )

    def test_expected_value_for_observable_with_multiple_qubit(
        self, dict_for_test_expected_value_with_compound_observable
    ):
        """[正常系]: 3粒子Qubit系に対する観測量の期待値"""
        dict_for_test = dict_for_test_expected_value_with_compound_observable
        expected_value = dict_for_test["observable"].expected_value(
            dict_for_test["qubits"]
        )
        assert round(expected_value, APPROX_DIGIT) == dict_for_test["expected_value"]

    def test_observation_for_multiple_qubit(
        self, dict_for_test_observation_with_compound_observable
    ):
        """[正常系]: von Neumann観測による3粒子Qubit系に対する観測"""
        dict_for_test = dict_for_test_observation_with_compound_observable

        # 観測量の第一成分の値と遷移後状態を期待する
        expected_result = dict_for_test["observable"].elements[0]

        # seedを固定してテストを可能にする
        random.seed(dict_for_test["randomize_seed"])

        # 観測実施
        observed_value = dict_for_test["observable"].observe(dict_for_test["qubits"])

        assert observed_value == expected_result["value"]
        assert np.all(
            np.round(
                dict_for_test["qubits"].amplitudes
                - expected_result["qubits"].amplitudes
            )
            == 0.0
        )

    def test_combine_observables(self, dict_for_test_combine_observables):
        """[正常系]: 2粒子Qubit系に対する観測量同士の結合"""
        dict_for_test = dict_for_test_combine_observables
        combined_observable = combine(
            dict_for_test["observable_group"][0], dict_for_test["observable_group"][1]
        )

        assert np.all(
            np.round(
                combined_observable.matrix - np.array(dict_for_test["expected_matrix"])
            )
            == 0.0
        )
