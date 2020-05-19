"""
quantum_simulator.base.qubitsモジュール内クラスのよく知られたオブジェクト群
"""

from quantum_simulator.base.qubits import Qubits, generalize
from quantum_simulator.major import pure_qubits

ZERO = Qubits([[1.0 + 0j, 0j], [0j, 0j]])
"""PureQubits: 0状態にあるQubit。｜0＞＜0｜"""

ONE = Qubits([[0j, 0j], [0j, 1.0 + 0j]])
"""PureQubits: 1状態にあるQubit。｜1＞＜1｜"""

PLUS = generalize(pure_qubits.PLUS)
"""PureQubits: プラス状態にあるQubit。｜+＞＜+｜"""

MINUS = generalize(pure_qubits.MINUS)
"""PureQubits: マイナス状態にあるQubit。｜-＞＜-｜"""

BELL_BASIS = [
    generalize(pure_qubits.BELL_BASIS.qubits_list[0]),
    generalize(pure_qubits.BELL_BASIS.qubits_list[1]),
    generalize(pure_qubits.BELL_BASIS.qubits_list[2]),
    generalize(pure_qubits.BELL_BASIS.qubits_list[3]),
]
"""List[Qubits]: Bell基底"""
