import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = 0

    def forward(self, data):
        self.loss = nn.functional.mse_loss(data, self.target)
        return data


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


class StyleTransfer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    def __get_style_model_and_losses(self, style_img, content_img):
        # normalization module
        normalization = Normalization(self.cnn_norm_mean, self.cnn_norm_std).to(self.device)

        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_{}".format(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu_{}".format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def transfer(self, content_img, style_img, noise=0.0,
                 num_steps=300, style_weight=1000000, content_weight=1):

        print("Style Transfer...")
        model, style_losses, content_losses = self.__get_style_model_and_losses(style_img, content_img)

        input_img = content_img.clone() + (torch.randn(content_img.data.size(), device=self.device) * noise)

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Iteration : {run[0]:4d} Style Loss : {style_score:4f} Content Loss: {content_score:4f}")

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
