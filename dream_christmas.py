import imageio
import numpy as np
import torch
import torchvision.transforms as transforms

from deep_dream.style_transfer import StyleTransfer
from deep_dream.deep_dream import DeepDream
from deep_dream.utils import tensor_to_image, load_image

import argparse

parser = argparse.ArgumentParser(description="Generate Style-Transfer image and apply DeepDream to produce GIF")

parser.add_argument("--style", required=True, help="File path to style image")
parser.add_argument("--content", required=True, help="File path to content image")
parser.add_argument("--style_iter", type=int, default=50, help="Style transfer iterations")
parser.add_argument("--style_noise", type=float, default=0.1, help="Magnitude of noise to add for style transfer")

parser.add_argument("--dream_iter", type=int, default=10, help="Dream per frame gradient ascent iterations")

parser.add_argument("--size", type=int, default=512, help="Output resolution")
parser.add_argument("--fps", type=int, default=24, help="Output GIF framerate")
parser.add_argument("--length", type=int, default=1, help="Output GIF length in seconds")
parser.add_argument("--output", default="output", help="Output filename")

args = parser.parse_args()


content_img  = load_image(args.content, args.size)
style_img    = load_image(args.style, args.size)
style_img = transforms.Resize(content_img.shape[-2:])(style_img)

style_transfer = StyleTransfer()
transfer_img = style_transfer.transfer(content_img, style_img, noise=args.style_noise, num_steps=args.style_iter)

# 'inception3b' 'inception4c', 'inception4d',
deep_dream = DeepDream(gradient_ascent_steps=args.dream_iter, layers_to_use=['inception4e'], step_size=0.01)
frames = [tensor_to_image(content_img), tensor_to_image(transfer_img), *map(tensor_to_image, deep_dream.dream_sequence(transfer_img, frames=(args.fps * args.length), rotate=0.1))]

def interpolate_frames(frames, index):

    index = (index * 0.9999) + 0.00001

    x0 = np.int32(np.floor(index))
    x1 = np.int32(x0 + 1)
    delta = (index - x0)

    print(index, x0, x1, delta)

    return ((frames[x0] * (1.0 - delta)) + (frames[x1] * delta)).astype(np.uint8)

indices = np.exp(np.linspace(0.0, 1.0, (args.fps * args.length))) * (len(frames) - 1)

imageio.mimwrite(f'{args.output}.gif', [*[interpolate_frames(frames, index) for index in indices],
                                        *[interpolate_frames(frames, index) for index in indices[::-1]]], fps=args.fps)
