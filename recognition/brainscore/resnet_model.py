import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from brainio.assemblies import DataAssembly
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
import os

BRAINIO_DIR = os.path.expanduser("~/.brainio/image_dicarlo_hvm-public")

def imagenet_preprocess(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

class ResNet18BrainModel(BrainModel):
    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()

        self._layer = 'layer4'
        self._preprocessing = imagenet_preprocess
        self._recording_layers = [self._layer]

        self._wrapper = PytorchWrapper(
            identifier='resnet18',
            model=self.model,
            preprocessing=imagenet_preprocess
        )

    def start_task(self, task: BrainModel.Task, **kwargs):
        pass

    def start_recording(self, target, *args, **kwargs):
        self._recording_layers = [self._layer]

    def look_at(self, stimuli, number_of_trials=1):
        filenames = stimuli['image_file_name'].values if 'image_file_name' in stimuli.columns else stimuli['filename'].values
        full_paths = [os.path.join(BRAINIO_DIR, name) for name in filenames]
        images = [self._preprocessing(Image.open(path).convert('RGB')) for path in full_paths]

        activations_dict = self._wrapper.get_activations(images, self._recording_layers)
        activation_array = activations_dict[self._recording_layers[0]]

        if activation_array.ndim > 2:
            activation_array = activation_array.reshape(activation_array.shape[0], -1)

        neuroid_ids = list(range(activation_array.shape[1]))
        stimulus_ids = list(stimuli['stimulus_id'])

        assembly = DataAssembly(
            data=activation_array,
            coords={
                'stimulus_id': ('presentation', stimulus_ids),
                'presentation': np.arange(len(stimulus_ids)),
                'neuroid': ('neuroid', neuroid_ids),
                'layer': ('neuroid', [self._recording_layers[0]] * len(neuroid_ids)),
                'object_name': ('presentation', list(stimuli['object_name'])) if 'object_name' in stimuli.columns else ('presentation', [''] * len(stimulus_ids)),
            },
            dims=['presentation', 'neuroid']
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
