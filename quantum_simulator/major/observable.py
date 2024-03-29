"""
quantum_simulator.base.observableモジュール内クラスのよく知られたオブジェクト群
"""

from ..base.observable import Observable, create_from_ons
from .pure_qubits import BELL_BASIS

BELL_OBSERVABLE = create_from_ons([0.0, 1.0, 2.0, 3.0], BELL_BASIS)
"""Observable: Bell基底による観測量"""

ZERO_OBSERVABLE = Observable([[0j, 0j], [0j, 0j]])
"""Observable: 零作用素(観測量)"""

IDENT_OBSERVABLE = Observable([[1.0 + 0j, 0j], [0j, 1.0 + 0j]])
"""Observable: 恒等作用素(観測量)"""

ZERO_PROJECTION = Observable([[1.0 + 0j, 0j], [0j, 0j]])
"""Observable: 標準基底に対する反真偽観測(i.e., ｜0＞＜0｜)"""

ONE_PROJECTION = Observable([[0j, 0j], [0j, 1.0 + 0j]])
"""Observable: 標準基底に対する真偽観測(i.e., ｜1＞＜1｜)"""
