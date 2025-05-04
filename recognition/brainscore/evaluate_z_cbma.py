from cornet_z_cbma_model import CORnetZZBrainModel
from brainscore_vision import benchmark_registry
import brainscore_vision.benchmarks.majajhong2015

model = CORnetZZBrainModel()

for area in ['V4', 'IT']:
    benchmark_id = f'MajajHong2015public.{area}-pls'
    benchmark = benchmark_registry[benchmark_id]()
    score = benchmark(model)
    print(f"{area} 层得分：", score.raw)
