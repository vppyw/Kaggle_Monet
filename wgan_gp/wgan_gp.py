# import module
import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
import math


# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0x6666)
workspace_dir = '.'

# prepare for CrypkoDataset

class MonetDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = MonetDataset(fnames, transform)
    return dataset

# Generator
class Generator(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 32 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 32 * 4 * 4),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 32, feature_dim * 16),
            self.dconv_bn_relu(feature_dim * 16, feature_dim * 8),
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),
            self.dconv_bn_relu(feature_dim * 2, feature_dim),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()

        self.l1 = nn.Sequential(
            self.conv_bn_lrelu(in_dim, feature_dim),
            self.conv_bn_lrelu(feature_dim, feature_dim),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),
            self.conv_bn_lrelu(feature_dim * 8, feature_dim * 16),
            nn.Conv2d(feature_dim * 16, 1, kernel_size=4, stride=1, padding=0),
        )
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.InstanceNorm2d(out_dim, affine = True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y

# setting for weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class TrainerGAN():
    def __init__(self, config):
        self.config = config

        self.G = Generator(self.config["z_dim"], self.config["g_dim"])
        self.D = Discriminator(3, self.config["d_dim"])

        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["d_lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["g_lr"], betas=(0.5, 0.999))

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')
        self.last_cktp = ''
        self.output_dir = os.path.join(self.config["workspace_dir"], 'output')

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).to(self.config["device"])

    def prepare_environment(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # update dir by time
        # time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, f'{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, f'{self.config["model_type"]}')
        self.output_dir = os.path.join(self.output_dir, f'{self.config["model_type"]}')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # create dataset by the above function
        dataset = get_dataset(os.path.join(self.config["dataset_dir"], 'monet_jpg'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

        # model preparation
        self.G = self.G.to(self.config["device"])
        self.D = self.D.to(self.config["device"])
        self.G.train()
        self.D.train()

    def gp(self, r_imgs, f_imgs):
        eta = torch.FloatTensor(r_imgs.size(0),1,1,1).uniform_(0,1)
        eta = eta.expand(*r_imgs.shape).to(self.config['device'])
        inter = eta * r_imgs + ((1 - eta) * f_imgs).to(self.config['device'])
        inter = Variable(inter, requires_grad = True)
        prob_inter = self.D(inter)
        gd = autograd.grad(outputs = prob_inter, inputs = inter,
                            grad_outputs = torch.ones(prob_inter.size()).to(self.config['device']),
                            create_graph = True, retain_graph = True)[0]
        return ((gd.norm(2, dim=1) - 1) ** 2).mean() * self.config['gp_lambda']

    def train(self):
        self.prepare_environment()

        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.to(self.config["device"])
                bs = imgs.size(0)

                z = Variable(torch.randn(bs, self.config["z_dim"])).to(self.config["device"])
                r_imgs = Variable(imgs).to(self.config["device"])
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).to(self.config["device"])
                f_label = torch.zeros((bs)).to(self.config["device"])


                # Discriminator forwarding
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                # Loss for discriminator
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + self.gp(r_imgs, f_imgs)

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"])).to(self.config["device"])
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)

                    # Loss for the generator.
                    loss_G = -torch.mean(self.D(f_imgs))

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            if epoch % self.config["log_step"] == 0:
                self.G.eval()
                f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
                filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
                torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
                logging.info(f'Save some samples to {filename}.')

            self.G.train()

            if (e+1) % self.config["log_step"] == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))
                self.last_cktp = os.path.join(self.ckpt_dir, f'G_{e}.pth') 

        logging.info('Finish training')

    def inference(self, G_path=None, n_generate=10, n_output=30, show=False):
        self.G.load_state_dict(torch.load(self.last_cktp))
        self.G.to(self.config["device"])
        self.G.eval()
        z = Variable(torch.randn(n_generate, self.config["z_dim"])).to(self.config["device"])
        imgs = (self.G(z).data + 1) / 2.0

        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], os.path.join(self.output_dir, f'{i+1}.jpg'))

config = {
    "model_type": "WGAN-GP-256",
    "batch_size": 16,
    "d_lr": 5e-5,
    "g_lr": 5e-4,
    "n_epoch": 1000,
    "log_step": 100,
    "n_critic": 2,
    "z_dim": 128,
    "d_dim": 64,
    "g_dim": 64,
    "clip_value": 0.005,
    "gp_lambda": 500,
    "workspace_dir": workspace_dir, # define in the environment setting
    "dataset_dir": "../Dataset/",
    "device":"cuda:2"
}

trainer = TrainerGAN(config)
trainer.train()
print(config)
trainer.inference()
