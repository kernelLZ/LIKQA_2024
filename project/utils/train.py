# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 17:14
# @Author  : BitYang
# @FileName: utils.py
# @Software: PyCharm
import argparse
from sre_parse import parse

import torch

from loss import get_loss_function
from train_dataloader import build_dataset
# from net import MobileNetKAN
from config import config
from mobile_mlp_net import MobileNetMerged


def select_optimizer_scheduler(network, config, train_loaders):
    optimizer = torch.optim.AdamW(
        lr=config.learning_rate,
        params=network.parameters(),
        weight_decay=config.weight_decay
    )

    warmup_iter = 0
    for train_loader in train_loaders.values():
        warmup_iter += int(config.warmup_epochs * len(train_loader))
    max_iter = int((config.num_epochs + config.l_num_epochs) * len(train_loader))

    lr_lambda = (
        lambda cur_iter: cur_iter / warmup_iter
        if cur_iter <= warmup_iter
        else 0.5 * (1 + torch.cos(torch.tensor(torch.pi * (cur_iter - warmup_iter) / (max_iter - warmup_iter))))
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


def train_epoch(network, loader, optimizer, scheduler, l2loss, plccloss, weights, color_space):
    cumu_loss = 0
    network.train()
    for _, data in enumerate(loader):
        images = data[color_space + '_Image'].cuda()
        labels = data['annotations'].cuda().float()

        outputs = network(images)

        outputs = outputs.view(outputs.size()[0], 1, 1)

        optimizer.zero_grad()

        NR_msel = l2loss(labels.flatten(), outputs.flatten())

        if torch.isnan(outputs).any() or torch.isnan(labels).any():
            # print(f"NaNs found in outputs or labels for task {task_id}")
            NR_crl = torch.tensor(0.0, device=outputs.device)
        else:
            NR_crl = plccloss(outputs.flatten()[None, :], labels.flatten()[None, :])

        loss = weights['NR_crl'] * NR_crl + weights['NR_msel'] * NR_msel

        loss.backward()
        optimizer.step()
        scheduler.step()

        cumu_loss += loss.item()
        print({"batch_loss": loss.item(), "NR_msel": NR_msel.item(), "NR_crl": NR_crl.item()})

    return cumu_loss / len(loader)


def validate_epoch(network, loader, l2loss, plccloss, weights, color_space):
    cumu_loss = 0
    network.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
            images = data[color_space + '_Image'].cuda()
            labels = data['annotations'].cuda().float()

            outputs = network(images)
            outputs = outputs.view(outputs.size()[0], 1, 1)

            loss = weights['NR_msel'] * l2loss(outputs.flatten(), labels.flatten()) + \
                   weights['NR_crl'] * plccloss(outputs.flatten()[None, :], labels.flatten()[None, :])

            cumu_loss += loss.item()

    return cumu_loss / len(loader)


def train(config):
    epochs = config.num_epochs

    train_loader = build_dataset(config.batch_size, config.csv_files, config.root_dirs)
    val_loader = build_dataset(config.batch_size, config.val_csv_file, config.val_root_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MobileNetMerged()
    # model = MobileNetKAN()
    print(model)
    model.to(device)

    optimizer, scheduler = select_optimizer_scheduler(model, config, {"train_loader": train_loader})
    l2loss = get_loss_function('l2')
    plccloss = get_loss_function('plcc')

    weights = {
        'NR_msel': config.NR_msel_weight,
        'NR_crl': config.NR_crl_weight
    }

    for epoch in range(epochs):
        avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, l2loss, plccloss,
            weights, config.color_space)
        avg_val_loss = validate_epoch(model, val_loader, l2loss, plccloss, weights, config.color_space)
        print({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})

        # Save model checkpoint
        model_name = f'./log/checkpoint_epoch_{epoch}.pt'
        torch.save(model.state_dict(), model_name)


def main():
    train(config)


if __name__ == "__main__":
    main()