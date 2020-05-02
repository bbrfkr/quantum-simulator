"""
量子テレポーテーションのシミュレーションコード
"""

from math import sqrt

import numpy as np
from numpy import linalg

from quantum_simulator.base import observable, qubits, transformer
from quantum_simulator.base.observable import Observable, ObservedBasis
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.base.transformer import UnitaryTransformer

# 初期状態の確率振幅
alpha = sqrt(0.3) + 0j
beta = sqrt(0.7) + 0j

# 転送したい初期状態を定義
initial_state = Qubits([alpha, beta])
print("##### 初期状態 #####")
initial_state.dirac_notation()
print()

# Bell基底の定義
bell_basis = [
    Qubits([[sqrt(0.5) + 0j, 0j], [0j, sqrt(0.5) + 0j]]),
    Qubits([[0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, 0j]]),
    Qubits([[0j, sqrt(0.5) + 0j], [-sqrt(0.5) + 0j, 0j]]),
    Qubits([[sqrt(0.5) + 0j, 0j], [0j, -sqrt(0.5) + 0j]]),
]
print("##### Qubit転送に利用するBell基底ベクトル #####")
bell_basis[0].dirac_notation()
print()

# 合成系の構成と全系ベクトルのDirac表記
whole_qubits = qubits.combine(initial_state, bell_basis[0])
print("##### 全系の状態ベクトル #####")
whole_qubits.dirac_notation()
print()

# 恒等観測量(作用素)の定義
standard_basis_2x2 = ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, 1.0 + 0j])])
identity_observable = Observable([1, 1], standard_basis_2x2)

# Bell基底によるAlice側の観測量を定義
print("##### Bell基底によるvon Neumann観測量 #####")
alice_observable = Observable([0.0, 1.0, 2.0, 3.0], ObservedBasis(bell_basis))
print(alice_observable)
print()

# 全系に対する観測量を定義
print("##### 全系に対するvon Neumann観測量 #####")
whole_observable = observable.combine(alice_observable, identity_observable)
print(whole_observable)
print()

# Aliceによる局所観測の実行
observed_value = whole_observable.observe(whole_qubits)
int_observed_value = int(observed_value)
print("##### 観測結果 #####")
print(observed_value)
print()

# 恒等変換(作用素)の定義
standard_basis_4x4 = ObservedBasis(
    [
        Qubits([[1 + 0j, 0j], [0j, 0j]]),
        Qubits([[0j, 1 + 0j], [0j, 0j]]),
        Qubits([[0j, 0j], [1 + 0j, 0j]]),
        Qubits([[0j, 0j], [0j, 1 + 0j]]),
    ]
)
identity_transformer = UnitaryTransformer(standard_basis_4x4, standard_basis_4x4)

# Bobが適用するユニタリ変換の定義
local_unitaries = [
    UnitaryTransformer(
        ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, 1.0 + 0j])]),
        ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, 1.0 + 0j])]),
    ),
    UnitaryTransformer(
        ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, 1.0 + 0j])]),
        ObservedBasis([Qubits([0j, 1.0 + 0j]), Qubits([1.0 + 0j, 0j])]),
    ),
    UnitaryTransformer(
        ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, 1.0 + 0j])]),
        ObservedBasis([Qubits([0j, -1.0 + 0j]), Qubits([1.0 + 0j, 0j])]),
    ),
    UnitaryTransformer(
        ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, 1.0 + 0j])]),
        ObservedBasis([Qubits([1.0 + 0j, 0j]), Qubits([0j, -1.0 + 0j])]),
    ),
]
whole_unitaries = [
    transformer.combine(identity_transformer, unitary) for unitary in local_unitaries
]
print("##### 観測結果に対応する局所ユニタリ変換 #####")
print(local_unitaries[int_observed_value])
print()

print("##### 観測結果に対応する全系のユニタリ変換 #####")
print(whole_unitaries[int_observed_value])
print()

# ユニタリ変換の適用
whole_unitaries[int_observed_value].operate(whole_qubits)

print("##### ユニタリ変換後の全系のベクトル #####")
whole_qubits.dirac_notation()
print()
