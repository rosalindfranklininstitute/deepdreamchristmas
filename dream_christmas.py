import torch
import imageio

from style_transfer.model.generator import Generator
from style_transfer.utils.img import tensor_to_image, load_image

import argparse

parser = argparse.ArgumentParser(description="Generate Style-Transfer image and apply DeepDream to produce GIF")

parser.add_argument("--style", required=True, help="File path to style image")
parser.add_argument("--content", required=True, help="File path to content image")
parser.add_argument("--style_iter", type=int, default=100, help="Style transfer iterations")
parser.add_argument("--style_noise", type=float, default=0.1, help="Magnitude of noise to add for style transfer")

parser.add_argument("--dream_iter", type=int, default=10, help="Dream per frame gradient ascent iterations")
parser.add_argument("--dream_layers", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])

parser.add_argument("--size", type=int, default=512, help="Output GIF resolution")
parser.add_argument("--fps", type=int, default=10, help="Output GIF framerate")
parser.add_argument("--output", default="output", help="Output GIF filename")

args = parser.parse_args()


def style_transfer(style_img, content_img, noise=0.0, iter=100):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed image is content image with some optional noise
    input_img = content_img.clone() + (torch.randn(content_img.data.size(), device=device) * noise)
    # Apply style transfer
    return Generator().run_style_transfer(content_img, style_img, input_img, num_steps=iter)


def deep_dream(seed_image):
    return []


style_img    = load_image(args.style,   args.size)
content_img  = load_image(args.content, args.size)
transfer_img = style_transfer(style_img, content_img, noise=args.style_noise, iter=args.style_iter)

frame = tensor_to_image(transfer_img)
frame.save(f'{args.output}.png')

frames = [frame, *deep_dream(transfer_img)]

imageio.mimwrite(f'{args.output}.gif', frames, fps=args.fps)
