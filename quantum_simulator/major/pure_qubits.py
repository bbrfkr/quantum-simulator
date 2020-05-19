"""
quantum_simulator.base.pure_qubitsモジュール内クラスのよく知られたオブジェクト群
"""

from math import sqrt

from quantum_simulator.base.pure_qubits import OrthogonalSystem, PureQubits

ZERO = PureQubits([1.0 + 0j, 0j])
"""PureQubits: 0状態にあるQubit。｜0＞"""

ONE = PureQubits([0j, 1.0 + 0j])
"""PureQubits: 1状態にあるQubit。｜1＞"""

PLUS = PureQubits([sqrt(0.5) + 0j, sqrt(0.5) + 0j])
"""PureQubits: プラス状態にあるQubit。｜+＞"""

MINUS = PureQubits([sqrt(0.5) + 0j, -sqrt(0.5) + 0j])
"""PureQubits: マイナス状態にあるQubit。｜-＞"""

BELL_BASIS = OrthogonalSystem(
    [
        PureQubits([sqrt(0.5) + 0j, 0j, 0j, sqrt(0.5) + 0j]),
        PureQubits([0j, sqrt(0.5) + 0j, sqrt(0.5) + 0j, 0j]),
        PureQubits([0j, sqrt(0.5) + 0j, -sqrt(0.5) + 0j, 0j]),
        PureQubits([sqrt(0.5) + 0j, 0j, 0j, -sqrt(0.5) + 0j]),
    ]
)
"""OrthogonalSystem: Bell基底"""
