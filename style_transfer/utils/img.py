import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch


def tensor_to_image(tensor):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    return unloader(image)


def load_image(path, size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = transforms.Compose([transforms.ToTensor(), transforms.Resize(size-1, max_size=size)])
    image = Image.open(path)
    image = loader(image).to(device, torch.float).unsqueeze(0)
    return image

