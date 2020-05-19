import numpy as np
import pytest

from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.time_evolution import (
    TimeEvolution,
    combine,
    compose,
    create_from_onb,
    multiple_combine,
    multiple_compose,
)
from quantum_simulator.base.utils import allclose


class TestTimeEvolution:
    """TimeEvolutionクラスと付随するメソッドのテスト"""

    def test_for_constructor(self, dict_for_test_unitary_constructor):
        """
        __init__メソッドの正常系テスト
        """
        target = dict_for_test_unitary_constructor["target"]
        unitary = TimeEvolution(target)

        matrix = unitary.matrix
        matrix_dim = unitary.matrix_dim
        ndarray = unitary.ndarray

        expected_matrix = np.array(dict_for_test_unitary_constructor["matrix"])
        expected_matrix_dim = dict_for_test_unitary_constructor["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_unitary_constructor["ndarray"])

        assert matrix_dim == expected_matrix_dim
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_invalid_unitary(self, dict_for_test_invalid_unitary):
        """異なるQubit数の粒子系に対する観測基底をもつユニタリ変換のテスト"""
        with pytest.raises(InitializeError):
            TimeEvolution(dict_for_test_invalid_unitary)

    def test_for_success_operate(self, test_for_success_operate):
        """
        operateメソッドの正常系テスト
        """

        unitary = test_for_success_operate["unitary"]
        target = test_for_success_operate["target"]

        operated_qubits = unitary.operate(target)

        expected_qubits = test_for_success_operate["expected_qubits"]
        assert allclose(operated_qubits.matrix, expected_qubits.matrix)

    def test_for_create_from_onb(self, dict_for_test_create_from_onb):
        """
        create_from_onbメソッドの正常系テスト
        """
        onb_0 = dict_for_test_create_from_onb["onb_0"]
        onb_1 = dict_for_test_create_from_onb["onb_1"]
        unitary = create_from_onb(onb_0, onb_1)

        matrix = unitary.matrix
        matrix_dim = unitary.matrix_dim
        ndarray = unitary.ndarray

        expected_matrix = np.array(dict_for_test_create_from_onb["matrix"])
        expected_matrix_dim = dict_for_test_create_from_onb["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_create_from_onb["ndarray"])

        assert matrix_dim == expected_matrix_dim
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_for_combine(self, dict_for_test_unitary_combine):
        """
        combineメソッドの正常系テスト
        """
        unitary_0 = dict_for_test_unitary_combine["unitary_0"]
        unitary_1 = dict_for_test_unitary_combine["unitary_1"]
        unitary = combine(unitary_0, unitary_1)

        matrix = unitary.matrix
        matrix_dim = unitary.matrix_dim
        ndarray = unitary.ndarray

        expected_matrix = np.array(dict_for_test_unitary_combine["matrix"])
        expected_matrix_dim = dict_for_test_unitary_combine["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_unitary_combine["ndarray"])

        assert matrix_dim == expected_matrix_dim
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_for_multiple_combine(self, dict_for_test_unitary_multiple_combine):
        """
        multiple_combineメソッドの正常系テスト
        """
        unitary_list = dict_for_test_unitary_multiple_combine["unitary_list"]
        unitary = multiple_combine(unitary_list)

        matrix = unitary.matrix
        matrix_dim = unitary.matrix_dim
        ndarray = unitary.ndarray

        expected_matrix = np.array(dict_for_test_unitary_multiple_combine["matrix"])
        expected_matrix_dim = dict_for_test_unitary_multiple_combine["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_unitary_multiple_combine["ndarray"])

        assert matrix_dim == expected_matrix_dim
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_for_compose(self, dict_for_test_unitary_compose):
        """
        combineメソッドの正常系テスト
        """
        unitary_0 = dict_for_test_unitary_compose["unitary_0"]
        unitary_1 = dict_for_test_unitary_compose["unitary_1"]
        unitary = compose(unitary_0, unitary_1)

        matrix = unitary.matrix
        matrix_dim = unitary.matrix_dim
        ndarray = unitary.ndarray

        expected_matrix = np.array(dict_for_test_unitary_compose["matrix"])
        expected_matrix_dim = dict_for_test_unitary_compose["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_unitary_compose["ndarray"])

        assert matrix_dim == expected_matrix_dim
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)

    def test_for_multiple_compose(self, dict_for_test_unitary_multiple_compose):
        """
        multiple_composeメソッドの正常系テスト
        """
        unitary_list = dict_for_test_unitary_multiple_compose["unitary_list"]
        unitary = multiple_compose(unitary_list)

        matrix = unitary.matrix
        matrix_dim = unitary.matrix_dim
        ndarray = unitary.ndarray

        expected_matrix = np.array(dict_for_test_unitary_multiple_compose["matrix"])
        expected_matrix_dim = dict_for_test_unitary_multiple_compose["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_unitary_multiple_compose["ndarray"])

        assert matrix_dim == expected_matrix_dim
        assert allclose(matrix, expected_matrix)
        assert allclose(ndarray, expected_ndarray)
        assert allclose(matrix.shape, expected_matrix.shape)
        assert allclose(ndarray.shape, expected_ndarray.shape)