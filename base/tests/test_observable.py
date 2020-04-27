import pytest

from base.conf import approx_digit
from base.error import CannotDistinguishError, NonOrthogonalError
from base.observable import Observable, ObserveBasis
from base.qubit import Qubit


class TestObserveBasis:
    def test_valid_observe_basis(self, orthogonal_qubits):
        """[正常系]: 直交したQubit同士で構成する観測基底"""
        observe_basis = ObserveBasis(orthogonal_qubits[0], orthogonal_qubits[1])
        assert observe_basis.qubits == orthogonal_qubits

    def test_invalid_observe_basis(self, non_orthogonal_qubits):
        """[異常系]: 直交しないQubit同士で観測基底を構成時、エラーとなること"""
        with pytest.raises(NonOrthogonalError):
            ObserveBasis(non_orthogonal_qubits[0], non_orthogonal_qubits[1])


class TestObservable:
    def test_valid_observable(self, valid_observed_value, observe_basis):
        """[正常系]: 状態識別可能な観測量"""
        observable = Observable(
            valid_observed_value[0], valid_observed_value[1], observe_basis
        )
        assert observable.observed_values[0] != observable.observed_values[1]

    def test_invalid_observable(self, invalid_observed_value, observe_basis):
        """[異常系]: 状態識別不能な観測量を構成時、エラーとなること"""
        with pytest.raises(CannotDistinguishError):
            Observable(
                invalid_observed_value[0], invalid_observed_value[1], observe_basis
            )

    def test_expected_value_for_observable_with_standard_basis(
        self, dict_for_test_expected_value
    ):
        """[正常系]: 標準基底に対する観測量の期待値"""
        dict_for_test = dict_for_test_expected_value
        expected_value = dict_for_test["observable"].expected_value(
            dict_for_test["qubit"]
        )
        assert round(expected_value, approx_digit) == dict_for_test["expected_value"]
