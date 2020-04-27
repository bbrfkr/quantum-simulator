import pytest

from base.error import NonOrthogonalError
from base.observable import Observable, ObserveBasis
from base.qubit import Qubit


class TestObserveBasis:
    def test_valid_observe_basis(self, orthogonal_qubits):
        """[正常系]: 直交したQubit同士の内積"""
        observe_basis = ObserveBasis(orthogonal_qubits[0], orthogonal_qubits[1])
        assert observe_basis.qubits == orthogonal_qubits

    def test_invalid_observe_basis(self, non_orthogonal_qubits):
        """[異常系]: 直交しないQubit同士の内積"""
        with pytest.raises(NonOrthogonalError):
            ObserveBasis(non_orthogonal_qubits[0], non_orthogonal_qubits[1])


class TestObservable:
    def test_expected_value_for_standard_projection(
        self, standard_projections, valid_qubit
    ):
        """[正常系]: 標準基底に対する射影観測の期待値"""
        target_qubit = Qubit(valid_qubit[0], valid_qubit[1])
        for index in [0, 1]:
            assert (
                standard_projections[index].expected_value(target_qubit)
                == abs(target_qubit.amplitudes[index]) ** 2
            )

    # def test_expected_value_for_observable_with_standard_basis(self, standard_observable, valid_qubit):
    #     """[正常系]: 標準基底に対する射影観測の期待値"""
    #     target_qubit = Qubit(valid_qubit[0], valid_qubit[1])
    #     observable = standard_observable
