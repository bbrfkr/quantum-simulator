# type: ignore
"""
BB84プロトコルのシミュレーションコード
qubit列を送って、偏光有無の古典情報から共有ビット列を作り出す
"""

import random

from quantum_simulator.base import observable, time_evolution
from quantum_simulator.base.observable import Observable
from quantum_simulator.base.qubits import specialize
from quantum_simulator.channel.channel import Channel
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
BITS_COUNT = 2


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
a_bits_int = candidate_bits_int
print(f"### 共有したいビット候補: {candidate_bits}")

# 2. aliceは、補助ビット列として、適当なビット列を1本作る
a_polarity_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
a_polarity_bits = convert_to_bits(a_polarity_bits_int, BITS_COUNT)

print(f"### aliceが適用する偏光状態のビット列: {a_polarity_bits}")

# 3. initializerを作成して初期量子ビットを作り出す
# ビット列の位の列と、文字列としての添え字の列が逆なので、そろえてからループを回す
# 観測量を結合する際、下位ビットに対する観測量が先に結合されるため、先に下位ビットを処理する
classical_unitaries = [
    NOT_GATE if int(bit) else IDENT_EVOLUTION for bit in reversed(candidate_bits)
]
classical_transformer = TimeEvolveTransformer(
    time_evolution.multiple_combine(classical_unitaries)
)

polarity_unitaries = [
    HADAMARD_GATE if int(bit) else IDENT_EVOLUTION for bit in reversed(a_polarity_bits)
]
polarity_transformer = TimeEvolveTransformer(
    time_evolution.multiple_combine(polarity_unitaries)
)

# registerの0番目はeve用、1番目はbob用
bb84_channel = Channel(BITS_COUNT, 2, [classical_transformer, polarity_transformer])
bb84_channel.initialize()

print("### 初期量子ビット列は以下の通り")
specialize(bb84_channel.states[0].qubits).dirac_notation()

# 4. eve(攻撃者)は、適当なビット列を一本作る
e_polarity_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
e_polarity_bits = convert_to_bits(e_polarity_bits_int, BITS_COUNT)
print(f"### eveが適用する偏光状態のビット列: {e_polarity_bits}")

# 5. eveは、aliceが送信した量子ビットに対して観測を行う
# まずは観測量を作る
e_observable_matrices = []

# ビット列の位の列と、文字列としての添え字の列が逆なので、そろえてからループを回す
# 観測量を結合する際、下位ビットに対する観測量が先に結合されるため、先に下位ビットを処理する
for i, bit in enumerate(reversed(e_polarity_bits)):
    whole_observable = None
    for j in range(BITS_COUNT):
        if i == j:
            if bit == "0":
                sub_observable = Observable(ONE.matrix * (2 ** i))
            else:
                sub_observable = Observable(MINUS.matrix * (2 ** i))
        else:
            sub_observable = IDENT_OBSERVABLE
        whole_observable = observable.combine(whole_observable, sub_observable)
    e_observable_matrices.append(whole_observable.matrix)

e_observable = Observable(sum(e_observable_matrices))
e_transformer = ObserveTransformer(e_observable)

print("!!! eveの観測 start !!!")
bb84_channel.transform(e_transformer, index=0)
print("!!! eveの観測 end !!!")
e_got_bits_int = round(bb84_channel.states[1].registers.get(0))
e_got_bits = convert_to_bits(e_got_bits_int, BITS_COUNT)
print(f"### eveが観測したビット列: {e_got_bits}")
print("### eveが観測した後の量子ビット列は以下の通り")
specialize(bb84_channel.states[1].qubits).dirac_notation()

# 6. bob(受信者)は、補助ビット列として、適当なビット列を1本作る
b_polarity_bits_int = random.randint(0, 2 ** BITS_COUNT - 1)
b_polarity_bits = convert_to_bits(b_polarity_bits_int, BITS_COUNT)

print(f"### bobが適用する偏光状態のビット列: {b_polarity_bits}")

# 7. bobは、eveを経由した量子ビットに対して観測を行う
# まずは観測量を作る
b_observable_matrices = []

# ビット列の位の列と、文字列としての添え字の列が逆なので、そろえてからループを回す
# 観測量を結合する際、下位ビットに対する観測量が先に結合されるため、先に下位ビットを処理する
for i, bit in enumerate(reversed(b_polarity_bits)):
    whole_observable = None
    for j in range(BITS_COUNT):
        if i == j:
            if bit == "0":
                sub_observable = Observable(ONE.matrix * (2 ** i))
            else:
                sub_observable = Observable(MINUS.matrix * (2 ** i))
        else:
            sub_observable = IDENT_OBSERVABLE
        whole_observable = observable.combine(whole_observable, sub_observable)
    b_observable_matrices.append(whole_observable.matrix)

b_observable = Observable(sum(b_observable_matrices))
b_transformer = ObserveTransformer(b_observable)

print("!!! bobの観測 start !!!")
bb84_channel.transform(b_transformer, index=1)
print("!!! bobの観測 end !!!")
b_got_bits_int = round(bb84_channel.states[2].registers.get(1))
b_got_bits = convert_to_bits(b_got_bits_int, BITS_COUNT)
print(f"### bobが観測したビット列: {b_got_bits}")
print("### bobが観測した後の量子ビット列は以下の通り")
specialize(bb84_channel.states[2].qubits).dirac_notation()

# aliceとbobは補助ビット同士で排他的論理和を取ったあと、反転する
# 1になったビットと同じ位の共有ビットをaliceとbobは鍵として利用する
# eveもその様子を見て、観測したビットを間引く
candidate_flags = ~(a_polarity_bits_int ^ b_polarity_bits_int)
a_shared_bits_int = 0
b_shared_bits_int = 0
e_shared_bits_int = 0
shared_bits_count = 0
for i in range(BITS_COUNT):
    if 1 & candidate_flags:
        a_target_bit = (a_bits_int & (2 ** i)) >> i
        b_target_bit = (b_got_bits_int & (2 ** i)) >> i
        e_target_bit = (e_got_bits_int & (2 ** i)) >> i
        a_shared_bits_int += a_target_bit * (2 ** shared_bits_count)
        b_shared_bits_int += b_target_bit * (2 ** shared_bits_count)
        e_shared_bits_int += e_target_bit * (2 ** shared_bits_count)
        shared_bits_count += 1
    candidate_flags >>= 1

a_shared_bits = convert_to_bits(a_shared_bits_int, shared_bits_count)
b_shared_bits = convert_to_bits(b_shared_bits_int, shared_bits_count)
e_shared_bits = convert_to_bits(e_shared_bits_int, shared_bits_count)
if shared_bits_count:
    print(f"### aliceが得た共有ビット列: {a_shared_bits}")
    print(f"### bobが得た共有ビット列: {b_shared_bits}")
    print(f"### eveが得た共有ビット列: {e_shared_bits}")

    if a_shared_bits != b_shared_bits:
        print("### eveが盗聴していることがaliceとbobにばれました！")
    else:
        print("### eveが盗聴していることにaliceとbobは気づくことができません。。。")
else:
    print("### aliceとbobは1ビットも共有できませんでした。。。")
