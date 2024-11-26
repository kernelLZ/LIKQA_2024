# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 16:00
# @Author  : BitYang
# @FileName: config.py
# @Software: PyCharm


class Config:
    # dataset
    csv_files = '/home/kernellz/Code/Paper/FIQA/train/train_label_score.txt'
    root_dirs = '/home/kernellz/Code/Paper/FIQA/train/img/'
    val_csv_file = '/home/kernellz/Code/Paper/FIQA/test/test_label_score.txt'
    val_root_dir = '/home/kernellz/Code/Paper/FIQA/test/img/'
    batch_size = 64
    num_epochs = 40
    color_space = "RGB"
    learning_rate = 5e-5
    weight_decay = 1e-4
    warmup_epochs = 5
    l_num_epochs = 10
    optimizer = "adamw"
    NR_msel_weight = 1.0
    NR_crl_weight = 1.0


config = Config()