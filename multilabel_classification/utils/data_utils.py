import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from skimage.io import imread
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
import pickle
from PIL import Image




class SurgeryDataset(Dataset):
    def __init__(self, data_dirs, labels, transform=None):
        self.data_dirs = data_dirs
        self.labels = [list(map(int, label_each[0])) for label_each in labels]#labels
        self.transform = transform

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, item):
        image = Image.open(self.data_dirs[item]).convert('RGB')
        label = torch.tensor(self.labels[item])

        image = self.transform(image)

        return image, label



def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    
    pkl_cholec40 = '../../CholecT80Classification/train_val_paths_labels_adjusted.pkl'
    pkl_endovis18 = '../../CholecT80Classification/miccai2018_train_val_paths_labels_adjusted.pkl'

    with open(pkl_cholec40, 'rb') as f:
        train_test_cholec = pickle.load(f)
    
    with open(pkl_endovis18, 'rb') as f:
        train_test_endovis = pickle.load(f)
    
    train_dirs = train_test_cholec[0] + train_test_endovis[0]
    train_labels = train_test_cholec[2] + train_test_endovis[2]
    valid_dirs = train_test_endovis[1]
    valid_labels = train_test_endovis[3]
    img_size = 224

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomApply(transforms=[transforms.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4084, 0.2551, 0.2535], [0.2266, 0.20202, 0.1962 ]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4084, 0.2551, 0.2535], [0.2266, 0.20202, 0.1962 ]),
    ])

    trainset = SurgeryDataset(train_dirs, train_labels, transform=transform_train)
    testset = SurgeryDataset(valid_dirs, valid_labels, transform=transform_test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader