"""
実験場
"""

from math import sqrt

from quantum_simulator.base import observable, pure_qubits, transformer
from quantum_simulator.base.observable import Observable, ObservedBasis
from quantum_simulator.base.pure_qubits import PureQubits
from quantum_simulator.base.qubits import Qubits, reduction, combine
from quantum_simulator.base.transformer import UnitaryTransformer


matrix_a = [[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]
matrix_b = [
    [[[0.5 + 0j, 0j], [0j, 0.5 + 0j]], [[0j, 0j], [0j, 0j]]],
    [[[0j, 0j], [0j, 0j]], [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]],
]

qubits_a = Qubits(density_array=matrix_a)
qubits_b = Qubits(density_array=matrix_b)
print(qubits_a)
print(qubits_b)
qubits_b.eigen_states[0].dirac_notation()


whole_qubits = combine(qubits_a, qubits_b)
print(whole_qubits)

qubits_c = reduction(whole_qubits, 1)
print(qubits_c)
print(reduction(qubits_c, 1))
