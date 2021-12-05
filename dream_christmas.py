import imageio

from deep_dream.style_transfer import StyleTransfer
from deep_dream.deep_dream import DeepDream
from deep_dream.utils import tensor_to_image, load_image

import argparse

parser = argparse.ArgumentParser(description="Generate Style-Transfer image and apply DeepDream to produce GIF")

parser.add_argument("--style", required=True, help="File path to style image")
parser.add_argument("--content", required=True, help="File path to content image")
parser.add_argument("--style_iter", type=int, default=100, help="Style transfer iterations")
parser.add_argument("--style_noise", type=float, default=0.1, help="Magnitude of noise to add for style transfer")

parser.add_argument("--dream_iter", type=int, default=10, help="Dream per frame gradient ascent iterations")
parser.add_argument("--dream_layers", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])

parser.add_argument("--size", type=int, default=512, help="Output resolution")
parser.add_argument("--fps", type=int, default=10, help="Output GIF framerate")
parser.add_argument("--output", default="output", help="Output filename")

args = parser.parse_args()


style_img    = load_image(args.style,   args.size)
content_img  = load_image(args.content, args.size)
transfer_img = StyleTransfer().transfer(style_img, content_img, noise=args.style_noise, num_steps=args.style_iter)

frame = tensor_to_image(transfer_img)
frame.save(f'{args.output}.png')

frame = tensor_to_image(DeepDream().dream(transfer_img))
frame.save(f'{args.output}-dream.png')

# frames = [frame]
# imageio.mimwrite(f'{args.output}.gif', frames, fps=args.fps)
