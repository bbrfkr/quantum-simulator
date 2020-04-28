from math import sqrt

import numpy as np

from base.qubits import Qubits, combine

# Qubit0: |0>
qubit0 = Qubits(np.array([1.0 + 0j, 0.0 + 0j]))

# Qubit1: |1>
qubit1 = Qubits(np.array([0.0 + 0j, 1.0 + 0j]))

# Compund Qubits: |0> otimes |1>
compound_qubits = combine(qubit0, qubit1)
print(compound_qubits)

# cannot recycle original qubits
print(f"{qubit0}{qubit1}")
