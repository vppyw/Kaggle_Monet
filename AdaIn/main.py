import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import numpy as np
import random
from tqdm import tqdm

import model
import dataset

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dnorm(x):
    mean = torch.Tensor([0.485, 0.456, 0.406])\
                .reshape(-1, 1, 1)\
                .to(x.get_device())
    std = torch.Tensor([0.299, 0.224, 0.225])\
                .reshape(-1, 1, 1)\
                .to(x.get_device())
    return torch.clamp(x * std + mean, 0, 1)

def mean(x):
    return sum(x) / len(x)

def main():
    same_seeds(0)
    config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'lambda': 1,
        'num_epoch': 80,
        'content_dir': '../dataset/photo_jpg',
        'style_dir': '../dataset/monet_jpg',
        'log_dir': './log',
        'output_dir': './output',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.299, 0.224, 0.225])
    ])

    adain = model.AdaIn().to(config['device'])
    monet_dataset = dataset.MonetDataset(config['content_dir'],
                                            config['style_dir'],
                                            tfm)
    dataloader = DataLoader(monet_dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=8)
    opt = torch.optim.AdamW(adain.parameters(),
                            lr=config['learning_rate'])

    for epoch in range(config['num_epoch']):
        content_losses = []
        style_losses = []
        for contents, styles in tqdm(dataloader, ncols=50):
            contents = contents.to(config['device'])
            styles = styles.to(config['device'])
            gen, gen_encode = adain(contents, styles)
            
            rec_embs = adain.encoder(gen, latent=True)
            style_embs = adain.encoder(styles, latent=True)
            
            content_loss = model.cal_content_loss(gen_encode,\
                                                    rec_embs[-1])
            style_loss = model.cal_style_loss(rec_embs,\
                                                style_embs)
            loss = content_loss + style_loss * config['lambda']
            
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
        log_fname = os.path.join(config['log_dir'], f'epoch_{epoch+1}.jpg')
        gen = dnorm(gen)
        torchvision.utils.save_image(gen, log_fname, nrow=8)        
        print(f'|content loss: {mean(content_losses):.5f}|style loss:| {mean(style_losses):.5f}')

if __name__ == '__main__':
    main()
