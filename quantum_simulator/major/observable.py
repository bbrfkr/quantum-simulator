"""
quantum_simulator.base.observableモジュール内クラスのよく知られたオブジェクト群
"""

from quantum_simulator.base.observable import Observable, create_from_ons
from quantum_simulator.major.pure_qubits import BELL_BASIS

BELL_OBSERVABLE = create_from_ons([0.0, 1.0, 2.0, 3.0], BELL_BASIS)
"""Observable: Bell基底による観測量"""

IDENT_OBSERVABLE = Observable([[1.0 + 0j, 0j], [0j, 1.0 + 0j]])
"""Observable: 恒等作用素(観測量)"""
