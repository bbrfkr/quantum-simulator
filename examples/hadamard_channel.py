"""
アダマールチャネルのシミュレーション
"""

from quantum_simulator.base.time_evolution import multiple_combine
from quantum_simulator.channel.channel import Channel
from quantum_simulator.channel.transformer import TimeEvolveTransformer
from quantum_simulator.major.time_evolution import HADAMARD_GATE


# アダマールチャネルの定義
class HadamardChannel(Channel):
    def calculate(self):
        qubit_count = self.qubit_count
        hadamard_gate = multiple_combine(
            [HADAMARD_GATE for index in range(qubit_count)]
        )
        self.transform(TimeEvolveTransformer(hadamard_gate))


test_channel = HadamardChannel(6, 6)
print(test_channel.apply(test_channel.apply(0, [0, 1, 2, 3, 4, 5]), [0, 1, 2, 3, 4, 5]))
