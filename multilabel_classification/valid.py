# coding=utf-8
#from __future__ import absolute_import, division, print_function

import random
import os
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tqdm import tqdm
from pathlib import Path

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
from sklearn.metrics import average_precision_score, accuracy_score
from arguments import get_args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    num_classes = 11
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    #model.load_from(np.load(args.pretrained_dir))
    model.load_state_dict(torch.load(args.pretrained_dir))
    model.to(args.device)

    return args, model

def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    val_outputs_tool_list = []
    val_labels_tool_list = []
    val_scores_tool_list = []
    val_loss_tool_list = []

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])


    logits = []
    for _, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            out = model(x)
            prob = torch.sigmoid(out)
            eval_loss = F.binary_cross_entropy(prob, y.float())
            eval_losses.update(eval_loss.item())
            val_outputs_tool_list.extend(prob.detach().cpu().numpy())
            val_labels_tool_list.extend(y.detach().cpu().numpy())
            scores_tool = torch.round(prob.data)
            val_scores_tool_list.extend(scores_tool.detach().cpu().numpy())

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
    val_pred = average_precision_score(np.array(val_labels_tool_list), np.array(val_outputs_tool_list), average = None)
    valid_mAP = np.nanmean(val_pred)
    print('Validation mAP:', valid_mAP)
    print(np.array(val_labels_tool_list).shape, np.array(val_scores_tool_list).shape)
    valid_acc = accuracy_score(np.array(val_labels_tool_list), np.array(val_scores_tool_list))
    print('Validation mAP:{}, Acc:{}'.format(valid_mAP, valid_acc))
    return eval_losses.avg, valid_mAP


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    args, model = setup(args)
    _, test_loader = get_loader(args)
    eval_losses, valid_mAP  = valid(args, model, test_loader)



if __name__ == "__main__":
    args = get_args()
    main(args)