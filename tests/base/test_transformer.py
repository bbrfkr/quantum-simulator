# import numpy as np
# import pytest

# from quantum_simulator.base import transformer
# from quantum_simulator.base.conf import APPROX_DIGIT
# from quantum_simulator.base.error import InitializeError
# from quantum_simulator.base.transformer import UnitaryTransformer


# class TestUnitaryTransformer:
#     """UnitaryTransformerクラスと付随するメソッドのテスト"""

#     def test_valid_unitary(self, valid_observed_basis_for_unitary):
#         """3粒子系の観測基底同士に対するユニタリ変換のテスト"""
#         unitary = UnitaryTransformer(
#             valid_observed_basis_for_unitary[0], valid_observed_basis_for_unitary[1]
#         )
#         assert unitary.pre_basis == valid_observed_basis_for_unitary[0]
#         assert unitary.post_basis == valid_observed_basis_for_unitary[1]

#     def test_invalid_unitary(self, invalid_observed_basis_for_unitary):
#         """異なるQubit数の粒子系に対する観測基底をもつユニタリ変換のテスト"""
#         with pytest.raises(InitializeError):
#             UnitaryTransformer(
#                 invalid_observed_basis_for_unitary[0],
#                 invalid_observed_basis_for_unitary[1],
#             )

#     def test_operation_by_unitary(self, dict_for_test_operation_of_unitary):
#         """ユニタリ変換によるQubit系へのオペレーションテスト"""
#         dict_for_test = dict_for_test_operation_of_unitary
#         dict_for_test["unitary"].operate(dict_for_test["target"])
#         assert np.all(
#             np.round(
#                 dict_for_test["target"].array - dict_for_test["expected_qubits"].array,
#                 APPROX_DIGIT,
#             )
#             == 0.0
#         )

#     def test_combine_of_unitaries(self, dict_for_test_combined_of_unitaries):
#         dict_for_test = dict_for_test_combined_of_unitaries
#         combined_unitary = transformer.combine(
#             dict_for_test["unitaries"][0], dict_for_test["unitaries"][1]
#         )
#         for index in range(len(combined_unitary.pre_basis.qubits_group)):
#             np.all(
#                 np.round(
#                     combined_unitary.pre_basis.qubits_group[index].array
#                     - dict_for_test["expected_unitary"]
#                     .pre_basis.qubits_group[index]
#                     .array,
#                     APPROX_DIGIT,
#                 )
#                 == 0.0
#             )
#         for index in range(len(combined_unitary.post_basis.qubits_group)):
#             np.all(
#                 np.round(
#                     combined_unitary.post_basis.qubits_group[index].array
#                     - dict_for_test["expected_unitary"]
#                     .post_basis.qubits_group[index]
#                     .array,
#                     APPROX_DIGIT,
#                 )
#                 == 0.0
#             )
