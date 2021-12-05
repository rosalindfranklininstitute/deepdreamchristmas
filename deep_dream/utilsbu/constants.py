import os
import enum
import numpy as np
import torch




class TRANSFORMS(enum.Enum):
    ZOOM = 0
    ZOOM_ROTATE = 1
    TRANSLATE = 2


class SupportedModels(enum.Enum):
    VGG16 = 0
    VGG16_EXPERIMENTAL = 1
    GOOGLENET = 2
    RESNET50 = 3
    ALEXNET = 4


class SupportedPretrainedWeights(enum.Enum):
    IMAGENET = 0
    PLACES_365 = 1


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')





