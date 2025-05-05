import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_interface import BrainModel
from brainio.assemblies import DataAssembly
from tqdm import tqdm
from cornet_z_cbma import CORnet_Z
# from cornet_z_vob import CORnet_Z

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
            identifier='cornet-zz-cbam-it',
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

    def start_task(self, task: BrainModel.Task, **kwargs): pass

    def start_recording(self, target, *args, **kwargs):
        self._recording_layers = [self._region_layer_map[target]]

    def look_at(self, stimuli, number_of_trials=1):
        filenames = stimuli['filename'].values
        full_paths = [os.path.join(self._image_path, name) for name in filenames]

        # 设置批大小（根据你显卡，建议 8～16）
        batch_size = 8
        activations_list = []

        for i in tqdm(range(0, len(full_paths), batch_size), desc="Batched Inference"):
            batch_paths = full_paths[i:i + batch_size]
            images = [self._preprocess(Image.open(p).convert('RGB')) for p in batch_paths]
            images = torch.stack(images)

            with torch.no_grad():
                activations_dict = self._wrapper.get_activations(images, layer_names=self._recording_layers)
            layer_name = self._recording_layers[0]
            activations = activations_dict[layer_name]

            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            if activations.ndim == 4:
                activations = activations.mean(axis=(2, 3))

            activations_list.append(activations)

        activation_array = np.concatenate(activations_list, axis=0)
        neuroid_ids = list(range(activation_array.shape[1]))
        stimulus_ids = stimuli['stimulus_id'].values

        return DataAssembly(
            data=activation_array,
            coords={
                'stimulus_id': ('presentation', stimulus_ids),
                'presentation': np.arange(len(stimulus_ids)),
                'neuroid': ('neuroid', neuroid_ids),
                'layer': ('neuroid', [layer_name] * len(neuroid_ids)),
                'region': ('neuroid', [layer_name.split('.')[0]] * len(neuroid_ids)),
                'object_name': ('presentation', list(stimuli['object_name'])),
            },
            dims=['presentation', 'neuroid'],
            name=layer_name
        )

    def extract_layers(self, image):
        return self._wrapper.extract_layers(image, layers=self._recording_layers)

    @property
    def visual_degrees(self): return 8
    @property
    def identifier(self): return self._wrapper.identifier
