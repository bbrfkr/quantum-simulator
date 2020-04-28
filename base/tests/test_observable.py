import random
from unittest.mock import patch

import pytest

from base.conf import approx_digit
from base.error import InitializeError
from base.observable import ObserveBasis
from base.qubits import Qubits


class TestObserveBasis:
    # 単一Qubitに対する観測基底のテスト
    def test_valid_observe_basis(self, orthogonal_qubits):
        """[正常系]: 直交した単一Qubit同士で構成する観測基底"""
        observe_basis = ObserveBasis(orthogonal_qubits)
        assert observe_basis.qubits_group == orthogonal_qubits

    def test_invalid_observe_basis(self, non_orthogonal_qubits):
        """[異常系]: 直交しない単一Qubit同士で観測基底を構成時、エラーとなること"""
        with pytest.raises(InitializeError) as error:
            ObserveBasis(non_orthogonal_qubits)
        assert "観測基底が直交しません" in str(error.value)

    # 複数Qubitに対する観測基底のテスト
    def test_valid_observe_basis_with_multiple_qubits(self, orthogonal_multiple_qubits):
        """[正常系]: 直交した複数Qubit同士で構成する観測基底"""
        observe_basis = ObserveBasis(orthogonal_multiple_qubits)
        assert observe_basis.qubits_group == orthogonal_multiple_qubits

    def test_invalid_observe_basis_with_non_orthogonal_multiple_qubits(
        self, non_orthogonal_multiple_qubits
    ):
        """[異常系]: 直交しない複数Qubit同士で観測基底を構成時、エラーとなること"""
        with pytest.raises(InitializeError) as error:
            ObserveBasis(non_orthogonal_multiple_qubits)
        assert "観測基底が直交しません" in str(error.value)


# class TestObservable:
#     def test_valid_observable(self, valid_observed_value, observe_basis):
#         """[正常系]: 状態識別可能な観測量"""
#         observable = Observable(
#             valid_observed_value[0], valid_observed_value[1], observe_basis
#         )
#         assert observable.observed_values[0] != observable.observed_values[1]

#     def test_invalid_observable(self, invalid_observed_value, observe_basis):
#         """[異常系]: 状態識別不能な観測量を構成時、エラーとなること"""
#         with pytest.raises(CannotDistinguishError):
#             Observable(
#                 invalid_observed_value[0], invalid_observed_value[1], observe_basis
#             )

#     def test_expected_value_for_observable_with_standard_basis(
#         self, dict_for_test_expected_value
#     ):
#         """[正常系]: 観測量の期待値"""
#         dict_for_test = dict_for_test_expected_value
#         expected_value = dict_for_test["observable"].expected_value(
#             dict_for_test["qubit"]
#         )
#         assert round(expected_value, approx_digit) == dict_for_test["expected_value"]

#     def test_observation_for_qubit(self, dict_for_test_observation):
#         """[正常系]: von Neumann観測によるQubitに対する観測"""
#         dict_for_test = dict_for_test_observation

#         # 観測量の第一成分の値と遷移後状態を期待する
#         expected_result = dict_for_test["observable"].observed_values[0]

#         # seedを固定してテストを可能にする
#         random.seed(1)

#         # 観測実施
#         observed_value = dict_for_test["observable"].observe(dict_for_test["qubit"])

#         assert observed_value == expected_result["value"]
#         assert dict_for_test["qubit"].amplitudes == expected_result["qubit"].amplitudes
