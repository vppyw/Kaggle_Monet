import os
import glob
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MonetDataset(Dataset):
    def __init__(self, content_dir, style_dir, tfm):
        self.contents = glob.glob(os.path.join(content_dir, '*.jpg'))
        self.styles = glob.glob(os.path.join(style_dir, '*.jpg'))
        self.len = len(self.contents)
        self.style_len = len(self.styles)
        self.tfm = tfm

    def __getitem__(self, idx):
        content = self.contents[idx]
        content = Image.open(content)
        content = self.tfm(content)
        style = self.styles[random.randint(0, self.style_len - 1)]
        style = Image.open(style)
        style = self.tfm(style)
        return content, style

    def __len__(self):
        return self.len
