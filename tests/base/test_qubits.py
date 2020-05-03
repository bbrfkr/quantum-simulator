import pytest

from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.pure_qubits import PureQubits
from quantum_simulator.base.qubits import Qubits


class TestQubits:
    """Qubitsクラスと付随するメソッドのテスト"""

    def test_no_input(self):
        """必須パラメータが無い場合"""
        with pytest.raises(InitializeError) as error:
            Qubits()
        assert "確率のリストとPureQubitsのリスト、もしくは密度行列のndarrayが必須です" in str(error.value)

    def test_valid_qubit_by_list(self, valid_qubit_list):
        """確率分布リストと純粋状態リストによる妥当な単一Qubit"""
        qubit = Qubits(
            valid_qubit_list["probabilities"], valid_qubit_list["pure_qubits"]
        )
