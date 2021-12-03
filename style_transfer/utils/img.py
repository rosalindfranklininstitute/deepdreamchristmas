import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torch import float


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def image_loader(image, imsize, device):
    loader = transforms.Compose(
        [transforms.Resize(imsize), transforms.ToTensor()]  # scale imported image
    )  # transform it into a torch tensor
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, float)


def resize2smallest(img1, img2):
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img1_area = img1.height * img1.width
    img2_area = img2.height * img2.width
    if img1_area > img2_area:
        img1 = img1.resize((img2.width, img2.height))

    else:
        img2 = img2.resize((img1.width, img1.height))
    return img1, img2
