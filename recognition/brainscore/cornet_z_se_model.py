import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_interface import BrainModel
from brainio.assemblies import DataAssembly
from tqdm import tqdm
from cornet_z_se import CORnet_Z  # 你自定义的模型

def imagenet_preprocess(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

class CORnetZZBrainModel(BrainModel):
    def __init__(self):
        model = CORnet_Z()
        model.eval()

        self._wrapper = PytorchWrapper(
            identifier='cornet-zz',
            model=model,
            preprocessing=imagenet_preprocess
        )

        self._region_layer_map = {
            'V1': 'V1.output',
            'V2': 'V2.output',
            'V4': 'V4.output',
            'IT': 'IT.output',
        }
        self._recording_layers = None
        self._image_path = os.path.expanduser("~/.brainio/image_dicarlo_hvm-public")
        self._preprocess = imagenet_preprocess

    def start_task(self, task: BrainModel.Task, **kwargs):
        pass

    def start_recording(self, target, *args, **kwargs):
        self._recording_layers = [self._region_layer_map[target]]

    def look_at(self, stimuli, number_of_trials=1):
        filenames = stimuli['filename'].values
        full_paths = [os.path.join(self._image_path, name) for name in filenames]
        images = [self._preprocess(Image.open(path).convert('RGB')) for path in tqdm(full_paths, desc="Preprocessing")]
        images = torch.stack(images)

        activations_dict = self._wrapper.get_activations(images, layer_names=self._recording_layers)
        layer_name = self._recording_layers[0]
        activations = activations_dict[layer_name]

        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()

        if isinstance(activations, np.ndarray) and activations.ndim == 2:
            activation_array = activations
        elif isinstance(activations, np.ndarray) and activations.ndim == 4:
            activation_array = activations.mean(axis=(2, 3))
        else:
            raise ValueError(f"Unexpected activation shape or type: {type(activations)}, shape: {getattr(activations, 'shape', None)}")

        neuroid_ids = list(range(activation_array.shape[1]))
        stimulus_ids = stimuli['stimulus_id'].values

        assembly = DataAssembly(
            data=activation_array,
            coords={
                'stimulus_id': ('presentation', stimulus_ids),
                'presentation': np.arange(len(stimulus_ids)),
                'neuroid': ('neuroid', neuroid_ids),
                'layer': ('neuroid', self._recording_layers * activation_array.shape[1]),
                'region': ('neuroid', [self._region_from_layer(l) for l in self._recording_layers] * activation_array.shape[1]),
                'object_name': ('presentation', list(stimuli['object_name'])),  # ✅ 非常关键，确保 benchmark 可用
            },
            dims=['presentation', 'neuroid']
        )
        assembly.name = layer_name
        return assembly

    def _region_from_layer(self, layer):
        return layer.split('.')[0]

    def extract_layers(self, image):
        return self._wrapper.extract_layers(image, layers=self._recording_layers)

    @property
    def visual_degrees(self):
        return 8

    @property
    def identifier(self):
        return self._wrapper.identifier
