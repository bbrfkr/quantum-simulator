"""
量子テレポーテーションのシミュレーションコード
"""

from math import sqrt

from quantum_simulator.base import observable, pure_qubits
from quantum_simulator.base.observable import Observable, observe
from quantum_simulator.base.pure_qubits import PureQubits, OrthogonalSystem
from quantum_simulator.base.qubits import Qubits, reduction, generalize, multiple_reduction, specialize

# 初期状態の確率振幅
alpha = sqrt(0.7) + 0j
beta = sqrt(0.3) + 0j

# 転送したい初期状態を定義
initial_qubit = PureQubits([alpha, beta])
print("##### 初期状態 #####")
initial_qubit.dirac_notation()
print()

# Bell基底の定義
bell_basis = [
    PureQubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
    PureQubits([[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]]),
    PureQubits([[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]]),
    PureQubits([[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]]),
]
print("##### Qubit転送に利用するBell基底ベクトル #####")
bell_basis[0].dirac_notation()
print()

# 合成系の構成と全系ベクトルのDirac表記
print(initial_qubit)
print(bell_basis[0])
whole_qubits = pure_qubits.combine(initial_qubit, bell_basis[0])
print("##### 全系の状態ベクトル #####")
whole_qubits.dirac_notation()
print()

# 恒等観測量(作用素)の定義
standard_basis_2x2 = OrthogonalSystem(
    [PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]
)
identity_observable = observable.create_from_ons([1, 1], standard_basis_2x2)

# Bell基底によるAlice側の観測量を定義
print("##### Bell基底によるvon Neumann観測量 #####")
alice_observable = observable.create_from_ons(
    [0.0, 1.0, 2.0, 3.0], OrthogonalSystem(bell_basis)
)
print(alice_observable)
print()

# 全系に対する観測量を定義
print("##### 全系に対するvon Neumann観測量 #####")
whole_observable = observable.combine(alice_observable, identity_observable)
print(whole_observable)
print()

# Aliceによる局所観測の実行
observed_value, converged_qubits = observe(whole_observable, generalize(whole_qubits))
int_observed_value = int(observed_value)
print("##### 観測結果 #####")
print(int_observed_value)
print()

# # 恒等変換(作用素)の定義
# standard_basis_4x4 = OrthogonalSystem(
#     [
#         PureQubits([[1 + 0j, 0j], [0j, 0j]]),
#         PureQubits([[0j, 1 + 0j], [0j, 0j]]),
#         PureQubits([[0j, 0j], [1 + 0j, 0j]]),
#         PureQubits([[0j, 0j], [0j, 1 + 0j]]),
#     ]
# )
# identity_transformer = UnitaryTransformer(standard_basis_4x4, standard_basis_4x4)

# Bobが適用するユニタリ変換の定義
# local_unitaries = [
#     UnitaryTransformer(
#         ObservedBasis([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]),
#         ObservedBasis([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]),
#     ),
#     UnitaryTransformer(
#         ObservedBasis([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]),
#         ObservedBasis([PureQubits([0j, 1.0 + 0j]), PureQubits([1.0 + 0j, 0j])]),
#     ),
#     UnitaryTransformer(
#         ObservedBasis([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]),
#         ObservedBasis([PureQubits([0j, -1.0 + 0j]), PureQubits([1.0 + 0j, 0j])]),
#     ),
#     UnitaryTransformer(
#         ObservedBasis([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, 1.0 + 0j])]),
#         ObservedBasis([PureQubits([1.0 + 0j, 0j]), PureQubits([0j, -1.0 + 0j])]),
#     ),
# ]
# whole_unitaries = [
#     transformer.combine(identity_transformer, unitary) for unitary in local_unitaries
# ]
# print("##### 観測結果に対応する局所ユニタリ変換 #####")
# print(local_unitaries[int_observed_value])
# print()

# print("##### 観測結果に対応する全系のユニタリ変換 #####")
# print(whole_unitaries[int_observed_value])
# print()


# # ユニタリ変換の適用
# whole_unitaries[int_observed_value].operate(whole_qubits)

print("##### Bobに送信されたユニタリ変換後のQubit #####")
output = multiple_reduction(converged_qubits, [0, 1])
print("### Matrix表示 ###")
print(output.matrix)
print("### Dirac表記表示 ###")
specialize(output).dirac_notation()
print()
