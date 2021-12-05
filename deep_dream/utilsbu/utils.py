import math
import numbers

import cv2 as cv
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from ..models.definitions.vggs import Vgg16, Vgg16Experimental
from ..models.definitions.googlenet import GoogLeNet
from ..models.definitions.resnets import ResNet50
from ..models.definitions.alexnet import AlexNet
from .constants import *


#
# Image manipulation util functions
#

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img





def build_image_name(config):
    input_name = 'rand_noise' if config['use_noise'] else config['input_name'].rsplit('.', 1)[0]
    layers = '_'.join(config['layers_to_use'])
    # Looks awful but makes the creation process transparent for other creators
    img_name = f'{input_name}_width_{config["img_width"]}_model_{config["model_name"]}_{config["pretrained_weights"]}_{layers}_pyrsize_{config["pyramid_size"]}_pyrratio_{config["pyramid_ratio"]}_iter_{config["num_gradient_ascent_iterations"]}_lr_{config["lr"]}_shift_{config["spatial_shift_size"]}_smooth_{config["smoothing_coefficient"]}.jpg'
    return img_name


def save_and_maybe_display_image(config, dump_img, name_modifier=None):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    # step1: figure out the dump dir location
    dump_dir = config['dump_dir']
    os.makedirs(dump_dir, exist_ok=True)

    # step2: define the output image name
    if name_modifier is not None:
        dump_img_name = str(name_modifier).zfill(6) + '.jpg'
    else:
        dump_img_name = build_image_name(config)

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img*255).astype(np.uint8)

    # step3: write image to the file system
    # ::-1 because opencv expects BGR (and not RGB) format...
    dump_path = os.path.join(dump_dir, dump_img_name)
    cv.imwrite(dump_path, dump_img[:, :, ::-1])

    # step4: potentially display/plot the image
    if config['should_display']:
        plt.imshow(dump_img)
        plt.show()

    return dump_path


#
# End of image manipulation util functions
#





# Didn't want to expose these to the outer API - too much clutter, feel free to tweak params here
def transform_frame(config, frame):
    h, w = frame.shape[:2]
    ref_fps = 30  # referent fps, the transformation settings are calibrated for this one

    if config['frame_transform'].lower() == TRANSFORMS.ZOOM.name.lower():
        scale = 1.04 * (ref_fps / config['fps'])  # Use this param to (un)zoom
        rotation_matrix = cv.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        frame = cv.warpAffine(frame, rotation_matrix, (w, h))

    elif config['frame_transform'].lower() == TRANSFORMS.ZOOM_ROTATE.name.lower():
        # Arbitrary heuristic keep the degree at 3 degrees/second and scale 1.04/second
        deg = 1.5 * (ref_fps / config['fps'])  # Adjust rotation speed (in [deg/frame])
        scale = 1.04 * (ref_fps / config['fps'])  # Use this to (un)zoom while rotating around image center
        rotation_matrix = cv.getRotationMatrix2D((w / 2, h / 2), deg, scale)
        frame = cv.warpAffine(frame, rotation_matrix, (w, h))

    elif config['frame_transform'].lower() == TRANSFORMS.TRANSLATE.name.lower():
        tx, ty = [2 * (ref_fps / config['fps']), 2 * (ref_fps / config['fps'])]
        translation_matrix = np.asarray([[1., 0., tx], [0., 1., ty]])
        frame = cv.warpAffine(frame, translation_matrix, (w, h))

    else:
        raise Exception('Transformation not yet supported.')

    return frame


def get_new_shape(config, base_shape, pyramid_level):
    SHAPE_MARGIN = 10
    pyramid_ratio = config['pyramid_ratio']
    pyramid_size = config['pyramid_size']
    exponent = pyramid_level - pyramid_size + 1
    new_shape = np.round(np.float32(base_shape)*(pyramid_ratio**exponent)).astype(np.int32)

    if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
        print(f'Pyramid size {config["pyramid_size"]} with pyramid ratio {config["pyramid_ratio"]} gives too small pyramid levels with size={new_shape}')
        print(f'Please change parameters.')
        exit(0)

    return new_shape




