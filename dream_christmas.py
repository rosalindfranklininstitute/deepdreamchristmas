import torch
from model.generator import Generator
from utils.img import imshow, image_loader, resize2smallest
import matplotlib.pyplot as plt
import argparse


def __option_parser():
    """
    Argument Parser
    Returns
    -------
    """
    parser = argparse.ArgumentParser(description="Generate StyleGan Image")
    parser.add_argument(
        "-s", "--style_image", required=True, help="File path to style image"
    )
    parser.add_argument(
        "-c", "--content_image", required=True, help="File path to content image"
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=300,
        help="Amount of iterations doing style transfer",
    )
    parser.add_argument(
        "-n", "--noise", action="store_true", help="Add noise, image will look crazy"
    )

    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = 512 if torch.cuda.is_available() else 128
    args = __option_parser()
    style_img, content_img = resize2smallest(args.style_image, args.content_image)

    style_img = image_loader(style_img, imsize, device)
    content_img = image_loader(content_img, imsize, device)
    if args.noise:
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        input_img = content_img.clone()

    plt.figure()
    imshow(content_img, title="Content Image")

    plt.figure()
    imshow(style_img, title="Style Image")

    gen = Generator()

    output = gen.run_style_transfer(
        content_img, style_img, input_img, num_steps=args.iterations
    )

    plt.figure()
    imshow(output, title="Output Image")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
