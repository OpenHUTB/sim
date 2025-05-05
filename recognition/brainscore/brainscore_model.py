import torch
from PIL import Image
from torchvision import transforms
from cornet import cornet_z
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_interface import BrainModel
import os
from brainio.assemblies import DataAssembly
import numpy as np

BRAINIO_DIR = os.path.expanduser("~/.brainio/image_dicarlo_hvm-public")

def imagenet_preprocess(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


class CORnetZBrainModel(BrainModel):
    def __init__(self):
        model = cornet_z()
        model.eval()
        self._preprocessing = imagenet_preprocess
        self._region_layer_map = {
            'V1': 'module.V1.output',
            'V2': 'module.V2.output',
            'V4': 'module.V4.output',
            'IT': 'module.IT.output',
        }
        self._recording_layers = None

        self._wrapper = PytorchWrapper(
            identifier='cornet-z',
            model=model,
            preprocessing=imagenet_preprocess
        )

    def start_task(self, task: BrainModel.Task, **kwargs):
        pass

    def start_recording(self, target, *args, **kwargs):
        if target not in self._region_layer_map:
            raise NotImplementedError(f'不支持的recording类型: {target}')
        self._recording_layers = [self._region_layer_map[target]]

    def look_at(self, stimuli, number_of_trials=1):
        print(stimuli.columns)
        filenames = stimuli['image_file_name'].values if 'image_file_name' in stimuli.columns else stimuli[
            'filename'].values
        full_paths = [os.path.join(os.path.expanduser('~/.brainio/image_dicarlo_hvm-public'), name) for name in
                      filenames]
        images = [self._preprocessing(Image.open(path).convert('RGB')) for path in full_paths]

        activations_dict = self._wrapper.get_activations(images, self._recording_layers)
        activation_array = activations_dict[self._recording_layers[0]]  # 提取对应层的数据
        if activation_array.ndim > 2:
            activation_array = activation_array.reshape(activation_array.shape[0], -1)

        # 构造 DataAssembly
        neuroid_ids = list(range(activation_array.shape[1]))
        stimulus_ids = list(stimuli['stimulus_id'])  # 注意：stimuli 必须是 StimulusSet，含 'stimulus_id' 列

        assembly = DataAssembly(
            data=activation_array,
            coords={
                'stimulus_id': ('presentation', stimulus_ids),
                'presentation': np.arange(len(stimulus_ids)),
                'neuroid': ('neuroid', neuroid_ids),
                'layer': ('neuroid', self._recording_layers * activation_array.shape[1]),
                'object_name': ('presentation', list(stimuli['object_name']))  # ✅ 添加这一行
            },
            dims=['presentation', 'neuroid'],
        )

        return assembly

    def extract_layers(self, image):
        return self._wrapper.extract_layers(image, layers=self._recording_layers)

    @property
    def visual_degrees(self):
        return 8

    @property
    def identifier(self):
        return self._wrapper.identifier
