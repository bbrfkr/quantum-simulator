from math import sqrt

from base.observable import Observable, ObserveBasis
from base.qubit import Qubit

# plus-minus 観測基底の定義 (|+>, |->) 
sign_basis = ObserveBasis(
    Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j), Qubit(sqrt(0.5) + 0j, -sqrt(0.5) + 0j)
)

# plus-minus 観測基底を用いた観測 (|+> -> 100, |-> -> -100) 
observable = Observable(100, -100, sign_basis)

# 観測対象Qubit
target = Qubit(1 + 0j, 0j)

print(f"pre-observed state: {target}")

# 観測実行
observed_value = observable.observe(target)

print(f"observed value: {observed_value}")
print(f"post-observed state: {target}")
