import numpy as np
import pytest

from quantum_simulator.base.conf import APPROX_DIGIT
from quantum_simulator.base.error import InitializeError
from quantum_simulator.base.qubits import Qubits


class TestQubits:
    """Qubitsクラスと付随するメソッドのテスト"""

    def test_no_input(self):
        """必須パラメータが無い場合"""
        with pytest.raises(InitializeError) as error:
            Qubits()
        assert "確率のリストとPureQubitsのリスト、もしくは密度行列のndarrayが必須です" in str(error.value)

    def test_valid_non_degrated_qubits_by_list(
        self, valid_orthogonal_non_degrated_qubits_list
    ):
        """確率分布リストと純粋状態リストによる妥当・非縮退のQubits"""
        qubits = Qubits(
            valid_orthogonal_non_degrated_qubits_list["probabilities"],
            valid_orthogonal_non_degrated_qubits_list["pure_qubits"],
        )
        assert np.all(
            np.round(
                np.array(qubits.eigen_values)
                - np.array(valid_orthogonal_non_degrated_qubits_list["probabilities"]),
                APPROX_DIGIT,
            )
            == 0.0
        )
        for index in range(len(qubits.eigen_values)):
            assert np.all(
                np.round(
                    np.array(qubits.eigen_states[index].array)
                    - np.array(
                        valid_orthogonal_non_degrated_qubits_list["pure_qubits"][
                            index
                        ].array
                    ),
                    APPROX_DIGIT,
                )
                == 0.0
            )

    def test_valid_degrated_qubits_by_list(self, valid_orthogonal_degrated_qubits_list):
        """確率分布リストと純粋状態リストによる妥当・縮退のQubits"""
        qubits = Qubits(
            valid_orthogonal_degrated_qubits_list["probabilities"],
            valid_orthogonal_degrated_qubits_list["pure_qubits"],
        )
        is_passed = False
        for result_index in range(len(qubits.eigen_values)):
            for expected_index in range(
                len(valid_orthogonal_degrated_qubits_list["probabilities"])
            ):
                if (
                    np.round(
                        qubits.eigen_values[result_index]
                        - valid_orthogonal_degrated_qubits_list["probabilities"][
                            expected_index
                        ],
                        APPROX_DIGIT,
                    )
                    == 0.0
                ):
                    if np.all(
                        np.round(
                            qubits.eigen_states[result_index].array
                            - valid_orthogonal_degrated_qubits_list["pure_qubits"][
                                expected_index
                            ].array,
                            APPROX_DIGIT,
                        )
                        == 0.0
                    ):
                        is_passed = True
        assert is_passed
