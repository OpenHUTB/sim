from alexnet_model import AlexNetBrainModel
from brainscore_vision import benchmark_registry
import brainscore_vision.benchmarks.majajhong2015  # 必须引入以注册 benchmark

model = AlexNetBrainModel()

for area in ['V4', 'IT']:
    benchmark_id = f'MajajHong2015public.{area}-pls'
    benchmark = benchmark_registry[benchmark_id]()
    score = benchmark(model)
    print(f"{area} 层得分：", score.raw)
