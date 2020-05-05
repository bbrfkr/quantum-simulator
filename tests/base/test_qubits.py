import numpy as np

from quantum_simulator.base.qubits import (
    Qubits,
    is_qubits_dim,
    resolve_arrays,
    resolve_eigen,
)
from quantum_simulator.base.utils import isclose, allclose


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
        assert np.allclose(matrix, expected_matrix)
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

        expected_eigen_values = dict_for_test_qubits_constructor["eigen_values"]
        expected_eigen_states = dict_for_test_qubits_constructor["eigen_states"]
        expected_matrix = np.array(dict_for_test_qubits_constructor["matrix"])
        expected_matrix_dim = dict_for_test_qubits_constructor["matrix_dim"]
        expected_ndarray = np.array(dict_for_test_qubits_constructor["ndarray"])
        expected_qubit_count = dict_for_test_qubits_constructor["qubit_count"]

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

    # def test_no_input(self):
    #     """[異常系]: 必須パラメータが無い場合"""
    #     with pytest.raises(InitializeError) as error:
    #         Qubits()
    #     assert "確率のリストとPureQubitsのリスト、もしくは密度行列のndarrayが必須です" in str(error.value)

    # def test_not_match_qubit_counts(self, not_match_qubit_counts_of_qubits_group):
    #     """[異常系]: Qubit数が異なるQubit群が与えられた場合"""
    #     with pytest.raises(InitializeError) as error:
    #         Qubits(
    #             not_match_qubit_counts_of_qubits_group["probabilities"],
    #             not_match_qubit_counts_of_qubits_group["pure_qubits"],
    #         )
    #     assert "与えられたQubit群のQubitの数が一致しません" in str(error.value)

    # def test_not_match_counts_of_qubits_and_probabilities(
    #     self, not_match_counts_of_qubits_and_probabilities
    # ):
    #     """[異常系]: 確率の数とQubit群の数が一致しない場合"""
    #     with pytest.raises(InitializeError) as error:
    #         Qubits(
    #             not_match_counts_of_qubits_and_probabilities["probabilities"],
    #             not_match_counts_of_qubits_and_probabilities["pure_qubits"],
    #         )
    #     assert "与えられたQubit群の数と確率の数が一致しません" in str(error.value)

    # def test_not_one_total_probability(self, not_one_total_probability):
    #     """[異常系]: 確率の数とQubit群の数が一致しない場合"""
    #     with pytest.raises(InitializeError) as error:
    #         Qubits(
    #             not_one_total_probability["probabilities"],
    #             not_one_total_probability["pure_qubits"],
    #         )
    #     assert "与えられた確率の総和が1となりません" in str(error.value)

    # def test_probabilities_of_negative_value(self, probabilities_of_negative_value):
    #     """[異常系]: 負の確率を含む場合"""
    #     with pytest.raises(InitializeError) as error:
    #         Qubits(
    #             probabilities_of_negative_value["probabilities"],
    #             probabilities_of_negative_value["pure_qubits"],
    #         )
    #     assert "値が負の確率が存在します" in str(error.value)

    # def test_not_corresponding_to_density_matrix(
    #     self, not_corresponding_to_density_matrix
    # ):
    #     """[異常系]: 与えられたarrayがQubit系の密度行列に対応しない場合"""
    #     with pytest.raises(InitializeError) as error:
    #         Qubits(density_array=not_corresponding_to_density_matrix)
    #     assert "与えられたlistはQubit系の密度行列に対応しません" in str(error.value)

    # def test_valid_qubits_by_orthogonal_list(self, valid_orthogonal_qubits_list):
    #     """[正常系]: 確率分布リストと直交する純粋状態リストによるQubits"""
    #     qubits = Qubits(
    #         valid_orthogonal_qubits_list["probabilities"],
    #         valid_orthogonal_qubits_list["pure_qubits"],
    #     )
    #     is_passed = False
    #     for result_index in range(len(qubits.eigen_values)):
    #         for expected_index in range(
    #             len(valid_orthogonal_qubits_list["probabilities"])
    #         ):
    #             if (
    #                 np.round(
    #                     qubits.eigen_values[result_index]
    #                     - valid_orthogonal_qubits_list[
    #                           "probabilities"
    #                       ][expected_index],
    #                     APPROX_DIGIT,
    #                 )
    #                 == 0.0
    #             ):
    #                 if np.all(
    #                     np.round(
    #                         qubits.eigen_states[result_index].array
    #                         - valid_orthogonal_qubits_list["pure_qubits"][
    #                             expected_index
    #                         ].array,
    #                         APPROX_DIGIT,
    #                     )
    #                     == 0.0
    #                 ):
    #                     is_passed = True
    #     assert is_passed
    #     assert qubits.qubit_count == valid_orthogonal_qubits_list["qubit_count"]
    #     assert qubits.matrix_dim == valid_orthogonal_qubits_list["matrix_dim"]

    # def test_valid_qubits_by_non_orthogonal_list(
    #     self, valid_non_orthogonal_qubits_list
    # ):
    #     """[正常系]: 確率分布リストと直交しない純粋状態リストによるQubits"""
    #     qubits = Qubits(
    #         valid_non_orthogonal_qubits_list["probabilities"],
    #         valid_non_orthogonal_qubits_list["pure_qubits"],
    #     )
    #     print(len(qubits.eigen_values))
    #     for expected_index in range(
    #         len(valid_non_orthogonal_qubits_list["expected_values"])
    #     ):
    #         is_passed = False
    #         for result_index in range(len(qubits.eigen_values)):
    #             if (
    #                 np.round(
    #                     qubits.eigen_values[result_index]
    #                     - valid_non_orthogonal_qubits_list["expected_values"][
    #                         expected_index
    #                     ],
    #                     APPROX_DIGIT,
    #                 )
    #                 == 0.0
    #             ):
    #                 if np.all(
    #                     np.round(
    #                         qubits.eigen_states[result_index].array
    #                         - valid_non_orthogonal_qubits_list["expected_states"][
    #                             expected_index
    #                         ].array,
    #                         APPROX_DIGIT,
    #                     )
    #                     == 0.0
    #                 ):
    #                     is_passed = True
    #         assert is_passed

    #     assert qubits.qubit_count == valid_non_orthogonal_qubits_list["qubit_count"]
    #     assert qubits.matrix_dim == valid_non_orthogonal_qubits_list["matrix_dim"]

    # def test_valid_qubits_by_density_matrix(self, valid_density_matrices):
    #     """[正常系]: 密度行列によるQubits"""
    #     qubits = Qubits(density_array=valid_density_matrices["density_matrix"])
    #     for expected_index in range(len(valid_density_matrices["expected_values"])):
    #         is_passed = False
    #         for result_index in range(len(qubits.eigen_values)):
    #             if (
    #                 np.round(
    #                     qubits.eigen_values[result_index]
    #                     - valid_density_matrices["expected_values"][expected_index],
    #                     APPROX_DIGIT,
    #                 )
    #                 == 0.0
    #             ):
    #                 if np.all(
    #                     np.round(
    #                         qubits.eigen_states[result_index].array
    #                         - valid_density_matrices["expected_states"][
    #                             expected_index
    #                         ].array,
    #                         APPROX_DIGIT,
    #                     )
    #                     == 0.0
    #                 ):
    #                     is_passed = True
    #         assert is_passed

    #     assert qubits.qubit_count == valid_density_matrices["qubit_count"]
    #     assert qubits.matrix_dim == valid_density_matrices["matrix_dim"]

    # def test_valid_qubits_by_density_arrays(self, valid_density_arrays):
    #     """[正常系]: 密度作用素によるQubits"""
    #     qubits = Qubits(density_array=valid_density_arrays["density_matrix"])
    #     for expected_index in range(len(valid_density_arrays["expected_values"])):
    #         is_passed = False
    #         for result_index in range(len(qubits.eigen_values)):
    #             if (
    #                 np.round(
    #                     qubits.eigen_values[result_index]
    #                     - valid_density_arrays["expected_values"][expected_index],
    #                     APPROX_DIGIT,
    #                 )
    #                 == 0.0
    #             ):
    #                 if np.all(
    #                     np.round(
    #                         qubits.eigen_states[result_index].array
    #                         - valid_density_arrays["expected_states"][
    #                             expected_index
    #                         ].array,
    #                         APPROX_DIGIT,
    #                     )
    #                     == 0.0
    #                 ):
    #                     is_passed = True
    #         assert is_passed

    #     assert qubits.qubit_count == valid_density_arrays["qubit_count"]
    #     assert qubits.matrix_dim == valid_density_arrays["matrix_dim"]
