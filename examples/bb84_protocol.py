"""
BB84プロトコルのシミュレーションコード
1byteのqubit列を送って、偏光有無の古典情報から共有ビット列を作り出す
"""

import random

import numpy as np

from quantum_simulator.base import observable, time_evolution
from quantum_simulator.base.observable import Observable
from quantum_simulator.base.qubits import specialize
from quantum_simulator.base.time_evolution import TimeEvolution
from quantum_simulator.channel.channel import Channel
from quantum_simulator.channel.initializer import Allocator, Initializer
from quantum_simulator.channel.transformer import (
    ObserveTransformer,
    TimeEvolveTransformer,
)
from quantum_simulator.major.observable import IDENT_OBSERVABLE
from quantum_simulator.major.qubits import MINUS, ONE
from quantum_simulator.major.time_evolution import (
    HADAMARD_GATE,
    IDENT_EVOLUTION,
    NOT_GATE,
)

# 設定
BITS_COUNT = 3


def convert_to_bits(target: int, bit_count: int) -> str:
    """
    非負値整数を、与えられたビット数で表現する
    """
    if target < 0:
        print("[ERROR]: 非負値を与えてください")
        raise
    if target > 2 ** bit_count - 1:
        print("[ERROR]: 与えられたビット数では対象を表現できません")
        raise
    return bin(target).split("b")[1].zfill(bit_count)


# 1. alice(送信者)は、まず適当なビット列を1本作る
candidate_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
candidate_bits = convert_to_bits(candidate_bits_int, BITS_COUNT)

print(f"共有したいビット候補: {candidate_bits}")

# 2. aliceとbob(受信者)は、それぞれ適当なビット列を1本作る
a_polarity_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
a_polarity_bits = convert_to_bits(a_polarity_bits_int, BITS_COUNT)
b_polarity_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
b_polarity_bits = convert_to_bits(b_polarity_bits_int, BITS_COUNT)

print(f"### aliceが適用する偏光状態のビット列: {a_polarity_bits}")
print(f"### bobが適用する偏光状態のビット列: {b_polarity_bits}")

# 2. initializerを作成して初期量子ビットを作り出す

classical_unitaries = [
    NOT_GATE if int(bit) else IDENT_EVOLUTION for bit in candidate_bits
]
classical_transformer = TimeEvolveTransformer(
    time_evolution.multiple_combine(classical_unitaries)
)

polarity_unitaries = [
    HADAMARD_GATE if int(bit) else IDENT_EVOLUTION for bit in a_polarity_bits
]
polarity_transformer = TimeEvolveTransformer(
    time_evolution.multiple_combine(polarity_unitaries)
)

# registerの0番目はeve用、1番目はbob用
bb84_channel = Channel(BITS_COUNT, 2, [classical_transformer, polarity_transformer])
bb84_channel.initialize()

print("### 初期量子ビットは以下の通り")
specialize(bb84_channel.states[0].qubits).dirac_notation()

# 3. eve(攻撃者)は、適当なビット列を一本作る
e_polarity_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
e_polarity_bits = convert_to_bits(e_polarity_bits_int, BITS_COUNT)
print(f"### eveが適用する偏光状態のビット列: {e_polarity_bits}")

# 4. eveは、aliceが送信した量子ビットに対して観測を行う
e_observable_matrices = []
for index, bit in enumerate(e_polarity_bits):
    if index == (BITS_COUNT - 1):
        if bit == "0":
            element = Observable(ONE.matrix * (2 ** (index)))
        else:
            element = Observable(MINUS.matrix * (2 ** index))
    else:
        element = IDENT_OBSERVABLE

    for j in range(BITS_COUNT - 1).__reversed__():
        if index == j:
            if bit == "0":
                o = Observable(ONE.matrix * (2 ** (index)))
            else:
                o = Observable(MINUS.matrix * (2 ** index))
        else:
            o = IDENT_OBSERVABLE
        element = observable.combine(element, o)

    e_observable_matrices.append(element.matrix)

e_observable = Observable(sum(e_observable_matrices))
e_transformer = ObserveTransformer(e_observable)

print(f"!!! eveの観測 start !!!")
bb84_channel.transform(e_transformer, index=0)
print(f"!!! eveの観測 end !!!")

print(round(bb84_channel.states[1].registers.get(0)))
print(bb84_channel.states[1].registers.get(0))
e_got_bits = convert_to_bits(round(bb84_channel.states[1].registers.get(0)), BITS_COUNT)
print(f"### eveが観測したビット: {e_got_bits}")
