# evaluate.py

# 必须添加这一句强制触发 benchmark 注册
import brainscore_vision.benchmarks  # ✅ 不可省略
import brainscore_vision.benchmarks.majajhong2015
from brainscore_vision import benchmark_registry
from brainscore_model import CORnetZBrainModel

model = CORnetZBrainModel()

# for area in ['V1', 'V2', 'V4', 'IT']:

for area in ['V4', 'IT']:
    benchmark_id = f'MajajHong2015public.{area}-pls'  # 注意点替换为点而非短横线
    benchmark = benchmark_registry[benchmark_id]()
    score = benchmark(model)
    print(f"{area} 层得分：", score.raw)
