import yaml
import torch
import shutil
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from dataloader.corrupt_loader import CorruptDataset
from models.common import reparameterize
from torchvision.utils import save_image
from corruptions import *
import matplotlib.pyplot as plt
import torchvision

import seaborn as sns
import numpy as np
import argparse
import os

from models.LUSR import *
from loss_procedures import *
from recon_save_procedures import *
from train_64 import get_assets


def main(cfg):
    arch = cfg['model']

    if arch == "DARLA":
        cfg['id'] = cfg['id'] + '_' + str(cfg['beta'])

    writer = SummaryWriter(f"runs/{arch}_{cfg['id']}")

    # Create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])

    eval_dataset = CorruptDataset(
        root=cfg['eval']['root'],
        corruption=cfg['eval']['corruptions'],
        intensity=cfg['eval']['intensities']
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'])

    # create model
    model, _, _ = get_assets(cfg)
    model.load_state_dict(torch.load(cfg['eval']['path']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Train
    corruptions = cfg['eval']['corruptions']
    num_corruptions = len(corruptions)
    num_points = len(eval_dataset)
    
    if IS_CLEAN:
        mu_l2_loss = [0] * num_corruptions
        sigma_l2_loss = [0] * num_corruptions
    else:
        size = num_corruptions + 1
        mu_l2_loss = np.zeros((size, size))
        sigma_l2_loss = np.zeros((size, size))

    model.eval()
    for j, imgs_list in enumerate(eval_loader):
        if IS_CLEAN:
            clean_mu = None
            clean_sigma = None
        else:
            mu_list = list()
            sigma_list = list()

        for i in range(len(imgs_list)):
            imgs = imgs_list[i].to(device)
            
            if arch == 'LUSR':
                mu, sigma, _, recon = model(imgs)
            else:
                mu, sigma, recon = model(imgs) 
            
            if IS_CLEAN:
                if i == 0:
                    clean_mu = mu
                    clean_sigma = sigma
                else:
                    mu_l2_loss[i-1] += torch.nn.functional.mse_loss(clean_mu, mu, reduction='sum')
                    sigma_l2_loss[i-1] += torch.nn.functional.mse_loss(clean_sigma, sigma, reduction='sum')
            else:
                mu_list.append(mu)
                sigma_list.append(sigma)
        
        if not IS_CLEAN:
            list_sz = len(mu_list)
            for i in range(list_sz):
                for k in range(list_sz):
                    mu_i = mu_list[i]
                    mu_k = mu_list[k]
                    sigma_i = sigma_list[i]
                    sigma_k = sigma_list[k]
                    mu_l2_loss[i][k] += torch.nn.functional.mse_loss(mu_i, mu_k, reduction='sum').item()
                    sigma_l2_loss[i][k] += torch.nn.functional.mse_loss(sigma_i, sigma_k, reduction='sum').item()
    
    if IS_CLEAN:
        for i in range(num_corruptions):
            corruption = corruptions[i]
            mu_l2_loss[i] /= num_points
            sigma_l2_loss[i] /= num_points
            
            writer.add_scalar(f"mu latent l2: {corruption}", mu_l2_loss[i])
            writer.add_scalar(f"sigma latent l2: {corruption}", sigma_l2_loss[i])

            print(f"Mu L2 Loss: {corruption}:{round(mu_l2_loss[i].item(),3)}")
            print(f"Sigma L2 Loss: {corruption}: {round(sigma_l2_loss[i].item(),3)}")
    else:
        corruptions = ['clean'] + corruptions
        mu = np.array(mu_l2_loss) / num_points
        sigma = np.array(sigma_l2_loss) / num_points
        
        hm_path = f"./heatmap/{arch}/{cfg['id']}"

        # Generate directories
        if not os.path.exists(hm_path):
            os.makedirs(hm_path)

        plt.figure()
        mu_hm = sns.heatmap(mu, xticklabels=corruptions, yticklabels=corruptions, annot=True, fmt=".3f")
        mu_hm.get_figure().savefig(f"{hm_path}/mu.png", dpi=400)

        plt.figure()
        sigma_hm = sns.heatmap(sigma, xticklabels=corruptions, yticklabels=corruptions, annot=True, fmt=".3f")
        sigma_hm.get_figure().savefig(f"{hm_path}/sigma.png", dpi=400)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/ddvae.yaml',
                        type=str, help='Path to yaml config file')
    parser.add_argument('--clean_only', default=False,
                        type=bool, help='Compare to clean latent only')
    args = parser.parse_args()

    IS_CLEAN = args.clean_only

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    main(cfg)