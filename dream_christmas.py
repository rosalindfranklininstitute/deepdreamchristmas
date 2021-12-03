import torch
import matplotlib.pyplot as plt

from style_transfer.model.generator import Generator
from style_transfer.utils.img import tensor_to_image, image_loader, resize2smallest

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
parser.add_argument("--output", default="output.gif", help="Output GIF filename")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load style and content images, resize them to the same size, then resize them both to the target resolution
style_img, content_img = resize2smallest(args.style, args.content)
style_img = image_loader(style_img, args.size, device)
content_img = image_loader(content_img, args.size, device)

def style_transfer(device, style_img, content_img, args):
    # Seed image is content image with some optional noise
    input_img = content_img.clone() + (torch.randn(content_img.data.size(), device=device) * args.style_noise)
    # Apply style transfer
    return Generator().run_style_transfer(content_img, style_img, input_img, num_steps=args.style_iter)

def deep_dream(device, seed_image, args):
    pass

transfer_img = style_transfer(device, style_img, content_img, args)


fig, ax = plt.subplots(1, 3)
plt.sca(ax[0])
plt.title('Style')
plt.imshow(tensor_to_image(style_img))
plt.sca(ax[1])
plt.title('Content')
plt.imshow(tensor_to_image(content_img))
plt.sca(ax[2])
plt.title('Transfer')
plt.imshow(tensor_to_image(transfer_img))
plt.show()
