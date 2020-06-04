"""
アダマールチャネルのシミュレーション
"""

import quantum_simulator.base.switch_cupy as switch_cupy
from quantum_simulator.base.time_evolution import multiple_combine
from quantum_simulator.channel.channel import Channel
from quantum_simulator.channel.transformer import TimeEvolveTransformer
from quantum_simulator.major.time_evolution import HADAMARD_GATE

if switch_cupy.is_cupy():
    import cupy as cp

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)


# アダマールチャネルの定義
class HadamardChannel(Channel):
    def calculate(self):
        qubit_count = self.qubit_count
        hadamard_gate = multiple_combine(
            [HADAMARD_GATE for index in range(qubit_count)]
        )
        self.transform(TimeEvolveTransformer(hadamard_gate))


test_channel = HadamardChannel(8, 8)
print(test_channel.apply(0, [0, 1, 2, 3, 4, 5, 6, 7]))
