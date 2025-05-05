from resnet_model import ResNet18BrainModel
from brainscore_vision import benchmark_registry
import brainscore_vision.benchmarks.majajhong2015

model = ResNet18BrainModel()

for area in ['V4', 'IT']:
    benchmark = benchmark_registry[f'MajajHong2015public.{area}-pls']()
    score = benchmark(model)
    print(f"{area} 层得分：", score.raw)
