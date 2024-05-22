import os
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam

from model.utils import get_dataloaders, viz_model
from model.model import YOLSOV1
from model.loss import *
import model.config as cfg

def train(train_loader, model, optimizer, loss_fn):
    # model.to(cfg.DEVICE)
    loop = train_loader
    losses = []

    for i, (image, label) in enumerate(loop):
        image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
        output = model(image)
        l = loss_fn(output, label)
        losses.append(l.item()) # fix
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # loop.set_postfix(loss=l.item())

    return np.mean(losses)

def val(val_loader, model, loss_fn):
    loop = val_loader
    total_losses = []
    coord_losses = []
    size_losses = []
    class_losses = []

    for i, (image, label) in enumerate(loop):
        image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
        with torch.no_grad():
            output = model(image)
            total_loss, coord_loss, size_loss, class_loss = loss_fn(output, label)
            total_losses.append(total_loss.item())
            coord_losses.append(coord_loss.item())
            size_losses.append(size_loss.item())
            class_losses.append(class_loss.item())

    return np.mean(total_losses), np.mean(coord_losses), np.mean(size_losses), np.mean(class_losses)

def plot_model(test_loader, model):
    for i, (image, label) in enumerate(test_loader):
        image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
        with torch.no_grad():
            output = model(image)
        print(image[0].shape, output[0].shape)
        viz_model(image[0], output[0])
        break

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
    else:
        print(f"No checkpoint found at '{filename}'")
        start_epoch = 0
    return start_epoch

def main():
    model = YOLSOV1(
        cfg.model_configs['origin_config'],
        3, 25, 8
    ).to(cfg.DEVICE)

    optimizer = Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = YOLSOV1Loss(25, 8)
    val_loss_fn = YOLSOV1ValLoss(25, 8)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

    for epoch in range(cfg.EPOCHS):
        train_loss = train(train_dataloader, model, optimizer, loss_fn)
        val_loss, val_coord_loss, val_size_loss, val_class_loss = val(val_dataloader, model, val_loss_fn)
        save_checkpoint(model, optimizer, epoch)

        print(f'Epoch {epoch + 1:3d}/{cfg.EPOCHS} | Train Loss: {train_loss:.10f} | '
              f'Validation Total Loss: {val_loss:.5f}, Coord Loss: {val_coord_loss:.5f}, Size Loss: {val_size_loss:.5f}, Class Loss: {val_class_loss:.5f}')
        if epoch % 2 == 0:
            plot_model(test_dataloader, model)

    del train_dataloader
    del val_dataloader
    del test_dataloader

    # Save model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()