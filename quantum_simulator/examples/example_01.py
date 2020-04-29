import os
import sys
from math import sqrt

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../src/")
)

import numpy as np

from base.observable import Observable, ObservedBasis
from base.qubits import Qubits

base = [0.0, 0.0, 0.0, 0.0]
basis = []
for index in range(4):
    np_base = base.copy()
    np_base[index] = 1.0
    np_base = np.array(np_base)
    np_base = np_base.reshape(2, 2)
    basis.append(list(np_base))

qubits_group = [Qubits(basis[index]) for index in range(4)]
observed_basis = ObservedBasis(qubits_group)
observable = Observable([1.0, 2.0, 3.0, 4.0], observed_basis)

target_qubits = Qubits([[sqrt(0.25), sqrt(0.25)], [sqrt(0.25), sqrt(0.25)]])
print("target qubits - dirac notation")
target_qubits.dirac_notation()
print()
print("target qubits - vector notation")
print(target_qubits)
print()
print("target qubits - array notation")
target_qubits.print_array()
print()
print("observable - matrix notation")
print(observable)
print()
print("observable - array notation")
observable.print_array()
print()
print("expected value of observation")
print(observable.expected_value(target_qubits))
print()
print("observed value")
print(observable.observe(target_qubits))
print()
print("post qubits - dirac notation")
target_qubits.dirac_notation()
