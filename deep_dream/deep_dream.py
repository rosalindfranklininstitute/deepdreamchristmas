"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
"""

import numbers
import numpy as np
import torch
from torch import nn
from torchvision import models

from collections import namedtuple


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LOWER_IMAGE_BOUND = np.array([-IMAGENET_MEAN / IMAGENET_STD], dtype=np.float32).reshape((1, -1, 1, 1))
UPPER_IMAGE_BOUND = np.array([(1 - IMAGENET_MEAN) / IMAGENET_STD], dtype=np.float32).reshape((1, -1, 1, 1))


class GoogLeNet(nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        googlenet = models.googlenet(pretrained=True, progress=show_progress).eval()

        self.layer_names = ['inception3b', 'inception4c', 'inception4d', 'inception4e']

        self.conv1 = googlenet.conv1
        self.maxpool1 = googlenet.maxpool1
        self.conv2 = googlenet.conv2
        self.conv3 = googlenet.conv3
        self.maxpool2 = googlenet.maxpool2

        self.inception3a = googlenet.inception3a
        self.inception3b = googlenet.inception3b
        self.maxpool3 = googlenet.maxpool3

        self.inception4a = googlenet.inception4a
        self.inception4b = googlenet.inception4b
        self.inception4c = googlenet.inception4c
        self.inception4d = googlenet.inception4d
        self.inception4e = googlenet.inception4e

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # todo: not sure why they are using this additional processing - made an issue
    #  https://discuss.pytorch.org/t/why-does-googlenet-additionally-process-input-via-transform-input/88865
    def transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self.transform_input(x)
        # N x 3 x 224 x 224
        x = self.conv1(x)
        conv1 = x
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        mp1 = x
        # N x 64 x 56 x 56
        x = self.conv2(x)
        conv2 = x
        # N x 64 x 56 x 56
        x = self.conv3(x)
        conv3 = x
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        mp2 = x

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        inception3a = x
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        inception3b = x
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        mp3 = x

        # N x 480 x 14 x 14
        x = self.inception4a(x)
        inception4a = x
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        inception4b = x
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        inception4c = x
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        inception4d = x
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        inception4e = x

        # Feel free to experiment with different layers.
        net_outputs = namedtuple("GoogLeNetOutputs", self.layer_names)
        out = net_outputs(inception3b, inception4c, inception4d, inception4e)
        return out


class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * np.sqrt(2 * np.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(self.device)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = nn.functional.conv2d

    def forward(self, input):
        input = nn.functional.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3


def circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = True
        return rolled


class DeepDream:

    def __init__(self, gradient_ascent_steps=10, smoothing_coefficient=0.5, step_size=0.09,
                 pyramid_size=1, pyramid_ratio=1.2, spatial_shift_size=32, layers_to_use=None):

        self.gradient_ascent_steps = gradient_ascent_steps
        self.smoothing_coefficient = smoothing_coefficient
        self.step_size = step_size
        self.pyramid_size = pyramid_size
        self.pyramid_ratio = pyramid_ratio
        self.spatial_shift_size = spatial_shift_size

        if layers_to_use is None:
            layers_to_use = ['relu4_3']
        self.layers_to_use = layers_to_use

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GoogLeNet(requires_grad=False, show_progress=True).to(self.device)
        self.layer_ids_to_use = [self.model.layer_names.index(layer_name) for layer_name in self.layers_to_use]

    def gradient_ascent(self, input_tensor, iteration):
        # loss.backward(layer) <- original implementation did it like this it's equivalent to MSE(reduction='sum')/2

        # Step 0: Feed forward pass
        out = self.model(input_tensor)

        # Step 1: Grab activations/feature maps of interest
        activations = [out[layer_id_to_use] for layer_id_to_use in self.layer_ids_to_use]

        # Step 2: Calculate loss over activations
        losses = []
        for layer_activation in activations:
            # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
            # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
            # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
            # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
            loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
            losses.append(loss_component)

        loss = torch.mean(torch.stack(losses))
        loss.backward()

        # Step 3: Process image gradients (smoothing + normalization)
        grad = input_tensor.grad.data

        # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
        # sigma is calculated using an arbitrary heuristic feel free to experiment
        sigma = ((iteration + 1) / self.gradient_ascent_steps) * 2.0 + self.smoothing_coefficient
        smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

        # Normalize the gradients (make them have mean = 0 and std = 1)
        # I didn't notice any big difference normalizing the mean as well - feel free to experiment
        g_std = torch.std(smooth_grad)
        g_mean = torch.mean(smooth_grad)
        smooth_grad = smooth_grad - g_mean
        smooth_grad = smooth_grad / g_std

        # Step 4: Update image using the calculated gradients (gradient ascent step)
        input_tensor.data += self.step_size * smooth_grad

        # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
        input_tensor.grad.data.zero_()
        input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


    def dream(self, image):

        is_chw = (image.shape[0] == 3)
        if is_chw:
            image = torch.moveaxis(image, 0, 2)

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = image.unsqueeze(0)
        image.requires_grad = True

        for iteration in range(self.gradient_ascent_steps):
            h_shift, w_shift = np.random.randint(-self.spatial_shift_size, self.spatial_shift_size + 1, 2)
            image = circular_spatial_shift(image, h_shift, w_shift)

            self.gradient_ascent(image, iteration)

            image = circular_spatial_shift(image, h_shift, w_shift, should_undo=True)

        image = (image.squeeze() * IMAGENET_STD) + IMAGENET_STD
        image = torch.clip(image, 0., 1.)

        if is_chw:
            image = torch.moveaxis(image, 2, 0)

        return image


#     def deep_dream_video_ouroboros(config):
#         """
#         Feeds the output dreamed image back to the input and repeat
#
#         Name etymology for nerds: https://en.wikipedia.org/wiki/Ouroboros
#
#         """
#         ts = time.time()
#         assert any([config['input_name'].lower().endswith(img_ext) for img_ext in SUPPORTED_IMAGE_FORMATS]), \
#             f'Expected an image, but got {config["input_name"]}. Supported image formats {SUPPORTED_IMAGE_FORMATS}.'
#
#         utils.print_ouroboros_video_header(config)  # print some ouroboros-related metadata to the console
#
#         img_path = utils.parse_input_file(config['input'])
#         # load numpy, [0, 1] range, channel-last, RGB image
#         # use_noise and consequently None value, will cause it to initialize the frame with uniform, [0, 1] range, noise
#         frame = None if config['use_noise'] else utils.load_image(img_path, target_shape=config['img_width'])
#
#         for frame_id in range(config['ouroboros_length']):
#             print(f'Ouroboros iteration {frame_id+1}.')
#             # Step 1: apply DeepDream and feed the last iteration's output to the input
#             frame = deep_dream_static_image(config, frame)
#             dump_path = utils.save_and_maybe_display_image(config, frame, name_modifier=frame_id)
#             print(f'Saved ouroboros frame to: {os.path.relpath(dump_path)}\n')
#
#             # Step 2: transform frame e.g. central zoom, spiral, etc.
#             # Note: this part makes amplifies the psychodelic-like appearance
#             frame = utils.transform_frame(config, frame)
#
#         video_utils.create_video_from_intermediate_results(config)
#         print(f'time elapsed = {time.time()-ts} seconds.')
#
# if __name__ == "__main__":
#
#     # Only a small subset is exposed by design to avoid cluttering
#     parser = argparse.ArgumentParser()
#
#     # Common params
#     parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name that will be used for dreaming", default='figures.jpg')
#     parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
#     parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])
#     parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
#                         help="Neural network (model) to use for dreaming", default=SupportedModels.VGG16_EXPERIMENTAL.name)
#     parser.add_argument("--pretrained_weights", choices=[pw.name for pw in SupportedPretrainedWeights],
#                         help="Pretrained weights to use for the above model", default=SupportedPretrainedWeights.IMAGENET.name)
#
#     # Main params for experimentation (especially pyramid_size and pyramid_ratio)
#     parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=4)
#     parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.8)
#     parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
#     parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.09)
#
#     # deep_dream_video_ouroboros specific arguments (ignore for other 2 functions)
#     parser.add_argument("--create_ouroboros", action='store_true', help="Create Ouroboros video (default False)")
#     parser.add_argument("--ouroboros_length", type=int, help="Number of video frames in ouroboros video", default=30)
#     parser.add_argument("--fps", type=int, help="Number of frames per second", default=30)
#     parser.add_argument("--frame_transform", choices=[t.name for t in TRANSFORMS],
#                         help="Transform used to transform the output frame and feed it back to the network input",
#                         default=TRANSFORMS.ZOOM_ROTATE.name)
#
#     # deep_dream_video specific arguments (ignore for other 2 functions)
#     parser.add_argument("--blend", type=float, help="Blend coefficient for video creation", default=0.85)
#
#     # You usually won't need to change these as often
#     parser.add_argument("--should_display", action='store_true', help="Display intermediate dreaming results (default False)")
#     parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)
#     parser.add_argument("--smoothing_coefficient", type=float, help='Directly controls standard deviation for gradient smoothing', default=0.5)
#     parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point instead of input image (default False)")
#     args = parser.parse_args()
#
#     # Wrapping configuration into a dictionary
#     config = dict()
#     for arg in vars(args):
#         config[arg] = getattr(args, arg)
#     config['dump_dir'] = OUT_VIDEOS_PATH if config['create_ouroboros'] else OUT_IMAGES_PATH
#     config['dump_dir'] = os.path.join(config['dump_dir'], f'{config["model_name"]}_{config["pretrained_weights"]}')
#     config['input_name'] = os.path.basename(config['input'])
#
#     # Create Ouroboros video (feeding neural network's output to it's input)
#     if config['create_ouroboros']:
#         deep_dream_video_ouroboros(config)
#
#     else:  # Create a static DeepDream image
#         print('Dreaming started!')
#         img = deep_dream_static_image(config, img=None)  # img=None -> will be loaded inside of deep_dream_static_image
#         dump_path = utils.save_and_maybe_display_image(config, img)
#         print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')

