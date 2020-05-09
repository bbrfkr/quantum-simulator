"""
quantum_simulator.base.transformerモジュール内クラスのよく知られたオブジェクト群
"""

from math import sqrt

from quantum_simulator.base.transformer import UnitaryTransformer

IDENT_TRANSFORMER = UnitaryTransformer([[1.0 + 0j, 0j], [0j, 1.0 + 0j]])
"""UnitaryTransformer: 恒等作用素(ユニタリ変換)"""

PAULI_MATRIX_X = UnitaryTransformer([[0j, 1.0 + 0j], [1.0 + 0j, 0j]])
"""UnitaryTransformer: Pauli行列 x"""

PAULI_MATRIX_Y = UnitaryTransformer([[0j, -1.0j], [1.0j, 0j]])
"""UnitaryTransformer: Pauli行列 y"""

PAULI_MATRIX_Z = UnitaryTransformer([[1.0 + 0j, 0j], [0j, -1.0 + 0j]])
"""UnitaryTransformer: Pauli行列 z"""

HADAMARD_GATE = UnitaryTransformer(
    [[sqrt(0.5) + 0j, sqrt(0.5) + 0j], [sqrt(0.5) + 0j, -sqrt(0.5) + 0j]]
)
"""UnitaryTransformer: Hadamardゲート"""

NOT_GATE = PAULI_MATRIX_X
"""UnitaryTransformer: NOTゲート"""

CNOT_GATE = UnitaryTransformer(
    [
        [1.0 + 0j, 0j, 0j, 0j],
        [0j, 1.0 + 0j, 0j, 0j],
        [0j, 0j, 0j, 1.0 + 0j],
        [0j, 0j, 1.0 + 0j, 0j],
    ]
)
"""UnitaryTransformer: CONTROLED-NOTゲート"""

SWAP_GATE = UnitaryTransformer(
    [
        [1.0 + 0j, 0j, 0j, 0j],
        [0j, 0j, 1.0 + 0j, 0j],
        [0j, 1.0 + 0j, 0j, 0j],
        [0j, 0j, 0j, 1.0 + 0j],
    ]
)
"""UnitaryTransformer: SWAPゲート"""

CCNOT_GATE = UnitaryTransformer(
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
"""UnitaryTransformer: CONTROLED-CONTROLED-NOT (Toffoli)ゲート"""
