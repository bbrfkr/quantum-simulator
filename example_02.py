from base.qubit_sequence import QubitSequence
from math import sqrt

print(QubitSequence(3, [
    sqrt(0.125) + 0j, sqrt(0.125) + 0j, sqrt(0.125) + 0j, sqrt(0.125) + 0j,
    sqrt(0.125) + 0j, sqrt(0.125) + 0j, sqrt(0.125) + 0j, sqrt(0.125) + 0j
    ]
))