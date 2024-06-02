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

    return [np.mean(total_losses), np.mean(coord_losses), np.mean(size_losses), np.mean(class_losses)]

def plot_model(test_loader, model, dir_path, epoch):
    for i, (image, label) in enumerate(test_loader):
        image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
        with torch.no_grad():
            output = model(image)
        viz_model(image[0], output[0], dir_path, epoch, i)
        break

def save_model_result(train_loss, val_losses, epoch, dir):
    with open(os.path.join(dir, 'train_losses.txt'), 'w') as f:
        losses = (f'Epoch {epoch + 1:3d}/{cfg.EPOCHS} | Train Total Loss: {train_loss:.7f} | Validation Total Loss: {val_losses[0]:.5f}, '
                  f'Coord Loss: {val_losses[1]:.5f}, Size Loss: {val_losses[2]:.5f}, Class Loss: {val_losses[3]:.5f} \n')
        f.write(losses)

def save_checkpoint(model, optimizer, epoch, dir, filename="checkpoint.pth.tar"):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(dir, filename))

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


def create_sequential_dir(base_path = cfg.OUTPUT_DIR, prefix="train"):
    """
    Create a new directory with a sequential number in the specified base path.

    Parameters:
        base_path (str): The path where the new directory should be created.
        prefix (str): The prefix for the new directory. Default is "train".

    Returns:
        new_dir_path (str): The path of the newly created directory.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # List existing directories in the base path
    existing_dirs = [d for d in os.listdir(base_path) if
                     os.path.isdir(os.path.join(base_path, d)) and d.startswith(prefix)]

    # Extract numbers from directory names
    existing_nums = [int(d.replace(prefix, '')) for d in existing_dirs if d.replace(prefix, '').isdigit()]

    # Find the next number
    next_num = max(existing_nums, default=-1) + 1

    # Create the new directory
    new_dir_name = f"{prefix}{next_num}"
    new_dir_path = os.path.join(base_path, new_dir_name)
    os.makedirs(new_dir_path)

    return new_dir_path

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

    dir_path = create_sequential_dir()

    for epoch in range(cfg.EPOCHS):
        if epoch % 10 == 0:
            plot_model(test_dataloader, model, dir_path, epoch)
        train_loss = train(train_dataloader, model, optimizer, loss_fn)
        val_losses = val(val_dataloader, model, val_loss_fn)
        save_checkpoint(model, optimizer, epoch, dir_path)
        save_model_result(train_loss, val_losses, epoch, dir_path)
        print(f'Epoch {epoch + 1:3d}/{cfg.EPOCHS} | Train Loss: {train_loss:.10f} | '
              f'Validation Total Loss: {val_losses[0]:.5f}, Coord Loss: {val_losses[1]:.5f}, Size Loss: {val_losses[2]:.5f}, Class Loss: {val_losses[3]:.5f}')


    del train_dataloader
    del val_dataloader
    del test_dataloader

    # Save model
    torch.save(model.state_dict(), os.path.join(dir_path, 'model.pth'))

if __name__ == '__main__':
    main()