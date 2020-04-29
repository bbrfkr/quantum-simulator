import os
import sys
from math import sqrt

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../src/")
)

import numpy as np

from base.observable import Observable, ObservedBasis
from base.qubits import Qubits
import base.qubits as qubits


initial_state = Qubits([1 + 0j, 0 + 0j])
bell_base = Qubits([
    [sqrt(0.5), 0], 
    [0, sqrt(0.5)]
])

whole_qubits = qubits.combine(initial_state, bell_base)
whole_qubits.dirac_notation()

bell_basis = [
    qubits.combine(Qubits([[sqrt(0.5), 0], [0, sqrt(0.5)]]), Qubits([1, 0])),
    qubits.combine(Qubits([[sqrt(0.5), 0], [0, sqrt(0.5)]]), Qubits([0, 1])),
    qubits.combine(Qubits([[0, sqrt(0.5)], [sqrt(0.5), 0]]), Qubits([1, 0])),
    qubits.combine(Qubits([[0, sqrt(0.5)], [sqrt(0.5), 0]]), Qubits([0, 1])),
    qubits.combine(Qubits([[0, sqrt(0.5)], [-sqrt(0.5), 0]]), Qubits([1, 0])),
    qubits.combine(Qubits([[0, sqrt(0.5)], [-sqrt(0.5), 0]]), Qubits([0, 1])),
    qubits.combine(Qubits([[sqrt(0.5), 0], [0, -sqrt(0.5)]]), Qubits([1, 0])),
    qubits.combine(Qubits([[sqrt(0.5), 0], [0, -sqrt(0.5)]]), Qubits([0, 1])),
]

alice_observable = Observable([1,1,2,2,3,3,4,4], ObservedBasis(bell_basis))
print(alice_observable)

result = alice_observable.observe(whole_qubits)
print(result)
whole_qubits.dirac_notation()

bob_basis = [
    qubits.combine(Qubits([[1, 0], [0, 0]]), Qubits([1, 0])),
    qubits.combine(Qubits([[0, 1], [0, 0]]), Qubits([1, 0])),
    qubits.combine(Qubits([[0, 0], [1, 0]]), Qubits([1, 0])),
    qubits.combine(Qubits([[0, 0], [0, 1]]), Qubits([1, 0])),
    qubits.combine(Qubits([[1, 0], [0, 0]]), Qubits([0, 1])),
    qubits.combine(Qubits([[0, 1], [0, 0]]), Qubits([0, 1])),
    qubits.combine(Qubits([[0, 0], [1, 0]]), Qubits([0, 1])),
    qubits.combine(Qubits([[0, 0], [0, 1]]), Qubits([0, 1])),
]
bob_observable = Observable([1,1,1,1,0,0,0,0], ObservedBasis(bob_basis))
print(bob_observable.expected_value(whole_qubits))
