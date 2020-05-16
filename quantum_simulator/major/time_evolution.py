"""
quantum_simulator.base.time_evolutionモジュール内クラスのよく知られたオブジェクト群
"""

from math import sqrt

from quantum_simulator.base.time_evolution import TimeEvolution

IDENT_EVOLUTION = TimeEvolution([[1.0 + 0j, 0j], [0j, 1.0 + 0j]])
"""TimeEvolution: 恒等作用素(時間発展)"""

PAULI_MATRIX_X = TimeEvolution([[0j, 1.0 + 0j], [1.0 + 0j, 0j]])
"""TimeEvolution: Pauli行列 x"""

PAULI_MATRIX_Y = TimeEvolution([[0j, -1.0j], [1.0j, 0j]])
"""TimeEvolution: Pauli行列 y"""

PAULI_MATRIX_Z = TimeEvolution([[1.0 + 0j, 0j], [0j, -1.0 + 0j]])
"""TimeEvolution: Pauli行列 z"""

HADAMARD_GATE = TimeEvolution(
    [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]
)
"""TimeEvolution: Hadamardゲート"""

NOT_GATE = PAULI_MATRIX_X
"""TimeEvolution: NOTゲート"""

CNOT_GATE = TimeEvolution(
    [
        [1.0 + 0j, 0j, 0j, 0j],
        [0j, 1.0 + 0j, 0j, 0j],
        [0j, 0j, 0j, 1.0 + 0j],
        [0j, 0j, 1.0 + 0j, 0j],
    ]
)
"""TimeEvolution: CONTROLED-NOTゲート"""

SWAP_GATE = TimeEvolution(
    [
        [1.0 + 0j, 0j, 0j, 0j],
        [0j, 0j, 1.0 + 0j, 0j],
        [0j, 1.0 + 0j, 0j, 0j],
        [0j, 0j, 0j, 1.0 + 0j],
    ]
)
"""TimeEvolution: SWAPゲート"""

CCNOT_GATE = TimeEvolution(
    [
        [1.0 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        [0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        [0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j, 0j],
        [0j, 0j, 0j, 1.0 + 0j, 0j, 0j, 0j, 0j],
        [0j, 0j, 0j, 0j, 1.0 + 0j, 0j, 0j, 0j],
        [0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j, 0j],
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j],
        [0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j, 0j],
    ]
)
"""TimeEvolution: CONTROLED-CONTROLED-NOT (Toffoli)ゲート"""
