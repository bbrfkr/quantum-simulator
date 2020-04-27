from math import sqrt
from base.qubit import Qubit
from base.observable import Observable, ObserveBasis

sign_basis = ObserveBasis(Qubit(sqrt(0.5) + 0j, sqrt(0.5) + 0j), Qubit(sqrt(0.5) + 0j, - sqrt(0.5) + 0j)) 
observable = Observable(100, -100, sign_basis)

target = Qubit(1 + 0j, 0j)

print(f"pre-observed state: {target}")

observed_value = observable.observe(target)

print(f"observed value: {observed_value}")
print(f"post-observed state: {target}")
