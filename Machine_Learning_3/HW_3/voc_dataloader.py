from torch.utils.data import DataLoader
from voc_dataset      import VOCDetection

import torch
import os


def default_collate(batch):
    label_ss, box_ss, image_s = [], [], []

    for sample in batch:
        image_s .append(sample[0])
        box_ss  .append(sample[1])
        label_ss.append(sample[2])

    return torch.stack(image_s), box_ss, label_ss

def get_train_dataloader(dataset_root_dir, annotation_file, batch_size, num_workers):
    train_dataset    = VOCDetection(dataset_root_dir, annotation_file )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn = default_collate, drop_last=True)
    return train_dataloader


def get_test_dataloader(dataset_root_dir, annotation_file, batch_size, num_workers):
    test_dataset    = VOCDetection( dataset_root_dir, annotation_file )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn = default_collate )
    return test_dataloader