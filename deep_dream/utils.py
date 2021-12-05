from PIL import Image
import torch
import torchvision.transforms as transforms


def tensor_to_image(tensor):
    image = tensor.squeeze(0).cpu().clone()
    return transforms.ToPILImage()(image)


def load_image(path, size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = transforms.Compose([transforms.ToTensor(), transforms.Resize(size-1, max_size=size)])
    image = Image.open(path)
    image = loader(image).to(device, torch.float).unsqueeze(0)
    return image

