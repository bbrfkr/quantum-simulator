import pytest
import numpy as np

from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.qubits import Qubits
from quantum_simulator.base.conf import APPROX_DIGIT


class TestQubits:
    """Qubitsクラスと付随するメソッドのテスト"""

    def test_no_input(self):
        """必須パラメータが無い場合"""
        with pytest.raises(InitializeError) as error:
            Qubits()
        assert "確率のリストとPureQubitsのリスト、もしくは密度行列のndarrayが必須です" in str(error.value)

    def test_valid_qubit_by_list(self, valid_orthogonal_non_degrated_qubit_list):
        """確率分布リストと純粋状態リストによる妥当・非縮退かつ単一Qubit"""
        qubit = Qubits(
            valid_orthogonal_non_degrated_qubit_list["probabilities"],
            valid_orthogonal_non_degrated_qubit_list["pure_qubits"],
        )
        assert np.all(
            np.round(
                np.array(qubit.eigen_values)
                - np.array(valid_orthogonal_non_degrated_qubit_list["probabilities"]),
                APPROX_DIGIT,
            )
            == 0.0
        )
        for index in range(len(qubit.eigen_values)):
            assert np.all(
                np.round(
                    np.array(qubit.eigen_states[index].amplitudes)
                    - np.array(
                        valid_orthogonal_non_degrated_qubit_list["pure_qubits"][
                            index
                        ].amplitudes
                    ),
                    APPROX_DIGIT,
                )
                == 0.0
            )
