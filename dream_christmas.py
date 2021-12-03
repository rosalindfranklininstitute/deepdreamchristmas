import torch
import matplotlib.pyplot as plt

from style_transfer.model.generator import Generator
from style_transfer.utils.img import imshow, image_loader, resize2smallest

import argparse

parser = argparse.ArgumentParser(description="Generate Style-Transfer image and apply DeepDream to produce GIF")
parser.add_argument("--size", type=int, default=512, help="Resolution")
parser.add_argument("-s", "--style", required=True, help="File path to style image")
parser.add_argument("-c", "--content", required=True, help="File path to content image")
parser.add_argument("-i", "--iterations", type=int, default=100, help="Style transfer iterations")
parser.add_argument("-n", "--noise", type=float, default=0.1, help="Magnitude of noise to add for style transfer")


args = parser.parse_args()


def style_transfer(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load style and content images, resize them to the same size, then resize them both to the target resolution
    style_img, content_img = resize2smallest(args.style, args.content)
    style_img   = image_loader(style_img, args.size, device)
    content_img = image_loader(content_img, args.size, device)

    # Seed image is content image with some optional noise
    input_img = content_img.clone() + (torch.randn(content_img.data.size(), device=device) * args.noise)

    plt.figure()
    imshow(content_img, title="Content Image")

    plt.figure()
    imshow(style_img, title="Style Image")

    output = Generator().run_style_transfer(content_img, style_img, input_img, num_steps=args.iterations)

    plt.figure()
    imshow(output, title="Output Image")

    plt.ioff()
    plt.show()

style_transfer(args)
