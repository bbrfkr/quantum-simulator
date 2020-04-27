from example_data import unitary_01
from base.qubit_sequence import QubitSequence
from base.generic_circuit import GenericCircuit
from math import sqrt

qubits = QubitSequence(3, [
    sqrt(0.4) + 0j, 0j, sqrt(0.125) + 0j, sqrt(0.125) + 0j,
    sqrt(0.125) + 0j, sqrt(0.125) + 0j, 0j, sqrt(0.1) + 0j
    ]
)
circuit = GenericCircuit(8, unitary_01)

print(qubits)
print(circuit)

print(circuit.operate(qubits))