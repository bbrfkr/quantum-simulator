"""
量子テレポーテーションのシミュレーションコード
"""

from math import sqrt

from quantum_simulator.base import observable, pure_qubits, time_evolution
from quantum_simulator.base.observable import observe
from quantum_simulator.base.pure_qubits import PureQubits
from quantum_simulator.base.qubits import generalize, multiple_reduction, specialize
from quantum_simulator.base.utils import around
from quantum_simulator.major.observable import IDENT_OBSERVABLE
from quantum_simulator.major.pure_qubits import BELL_BASIS
from quantum_simulator.major.time_evolution import (
    IDENT_EVOLUTION,
    PAULI_MATRIX_X,
    PAULI_MATRIX_Z,
)

# 初期状態の確率振幅
alpha = sqrt(0.7) + 0j
beta = sqrt(0.3) + 0j

# 転送したい初期状態を定義
input_qubit = PureQubits([alpha, beta])
print("##### 初期の入力状態 #####")
print()

print("### Matrix表示 ###")
print(input_qubit.projection_matrix)
print()

print("### Dirac表記表示 ###")
input_qubit.dirac_notation()
print()

# 転送用Bell状態の表示
print("##### Qubit転送に利用するBell基底ベクトル #####")
BELL_BASIS.qubits_list[0].dirac_notation()
print()

# 合成系の構成と全系ベクトルのDirac表記
whole_qubits = pure_qubits.combine(input_qubit, BELL_BASIS.qubits_list[0])
print("##### 全系の状態ベクトル #####")
whole_qubits.dirac_notation()
print()

# Bell基底によるAlice側の観測量を定義
print("##### Bell基底によるvon Neumann観測量 #####")
alice_observable = observable.create_from_ons([0.0, 1.0, 2.0, 3.0], BELL_BASIS)
print(alice_observable)
print()

# 全系に対する観測量を定義
print("##### 全系に対するvon Neumann観測量 #####")
whole_observable = observable.combine(alice_observable, IDENT_OBSERVABLE)
print(whole_observable)
print()

# Aliceによる局所観測の実行
observed_value, converged_qubits = observe(whole_observable, generalize(whole_qubits))
int_observed_value = int(around(observed_value))
print("##### 観測結果 #####")
print(int_observed_value)
print()

# Bobが適用するユニタリ変換の定義
local_unitaries = [
    IDENT_EVOLUTION,
    PAULI_MATRIX_X,
    time_evolution.compose(PAULI_MATRIX_X, PAULI_MATRIX_Z),
    PAULI_MATRIX_Z,
]
bob_unitary = local_unitaries[int_observed_value]
bob_qubit = multiple_reduction(converged_qubits, [0, 1])

print("##### 観測直後の出力状態 #####")
print(bob_qubit)
print()

print("##### 観測結果に対応するユニタリ変換 #####")
print(bob_unitary)
print()

# ユニタリ変換の適用
output_qubit = bob_unitary.operate(bob_qubit)

print("##### 最終的な出力状態 #####")
print()

print("### Matrix表示 ###")
print(output_qubit.matrix)
print()

print("### Dirac表記表示 ###")
specialize(output_qubit).dirac_notation()
print()
