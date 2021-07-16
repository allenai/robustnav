import torch
import torchvision

def load_train_frames(cfg):
    train_dataset = torchvision.datasets.ImageFolder(
        root=cfg['training']['root'],
        transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['n_workers'],
        shuffle=True
    )

    return train_loader


def load_eval_frames(cfg):
    dataset = torchvision.datasets.ImageFolder(
        root=cfg['training']['eval'],
        transform=torchvision.transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['n_workers'],
        shuffle=True
    )

    return loader