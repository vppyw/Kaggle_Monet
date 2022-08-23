import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg19(
                        weights=models.VGG19_Weights.IMAGENET1K_V1
                    ).features
        self.relu1 = features[:2]       # vgg19: relu1
        self.relu2 = features[2:7]      # vgg19: relu2
        self.relu3 = features[7:12]     # vgg19: relu3
        self.relu4 = features[12:21]    # vgg19: relu4

        for i in range(1, 5):
            for param in getattr(self, f'relu{i}').parameters():
                param.requires_grad = False

    def forward(self, x, latent=False):
        h1 = self.relu1(x)
        h2 = self.relu2(h1)
        h3 = self.relu3(h2)
        h4 = self.relu4(h3)
        if latent:
            return h1, h2, h3, h4
        return h4

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=8,
                        kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3,
                        kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.cnn(x)

def cal_mean_std(x, esp=1e-12):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + esp
    return mean, std

class AdaIn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, content, style):
        content_encode = self.encoder(content)
        style_encode = self.encoder(style)

        content_mean, content_std = cal_mean_std(content_encode)
        style_mean, style_std = cal_mean_std(style_encode)
        gen_encode = style_std \
                    * (content_encode - content_mean) \
                    / content_std \
                    + style_mean
        gen = self.decoder(gen_encode)

        return gen, gen_encode

def cal_content_loss(gen_encode, rec_encode):
    return F.mse_loss(rec_encode, gen_encode)

def cal_style_loss(gen_encode, style_encode):
    loss = 0
    for gen, style in zip(gen_encode, style_encode):
        gen_mean, gen_std = cal_mean_std(gen)
        style_mean, style_std = cal_mean_std(style)
        loss += F.mse_loss(gen_mean, style_mean) + \
                F.mse_loss(gen_std, style_std)
    return loss
