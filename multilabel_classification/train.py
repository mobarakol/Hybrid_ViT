# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
#from utils.dist_util import get_world_size
import torch.nn.functional as F

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, accuracy_score
from arguments import get_args

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    #num_classes = 10 if args.dataset == "cifar10" elif 100
    num_classes = 11
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    train_outputs_tool_list = []
    train_labels_tool_list = []
    train_scores_tool_list = []
    train_loss_tool_list = []

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)
            eval_loss = F.binary_cross_entropy(prob, y.float())
            #eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            train_outputs_tool_list.extend(prob.detach().cpu().numpy())
            train_labels_tool_list.extend(y.detach().cpu().numpy())


        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    train_pred = average_precision_score(np.array(train_labels_tool_list), np.array(train_outputs_tool_list), average = None)
    valid_mAP = np.nanmean(train_pred)
    print(np.array(train_labels_tool_list).shape, np.array(train_scores_tool_list).shape)
    #train_acc = accuracy_score(np.array(train_labels_tool_list), np.array(train_scores_tool_list))
    train_mean_loss = np.mean(np.array(train_loss_tool_list))
    print('Validation mAP:', valid_mAP)

    return eval_losses.avg, valid_mAP


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare datase
    train_loader, test_loader = get_loader(args)
    print('Sample size: overal train={}, valid={}'.format(len(train_loader.dataset), len(test_loader.dataset)))
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    print('2mobarak set_seed(args) set_seed(args) set_seed(args)')
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    print('3mobarak set_seed(args) set_seed(args) set_seed(args)')
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    best_global_step = 0
    criterion_tool = nn.MultiLabelSoftMarginLoss()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            logits = model(x)
            prob = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(prob, y.float())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    val_loss, mAP = valid(args, model, writer, test_loader, global_step)
                    if best_acc < mAP:
                        save_model(args, model)
                        best_acc = mAP
                        best_global_step = global_step
                    print('Cuurent mAP:{}, Best mAP:{}, Global Step(curr/best):({}/{}), lr: {}'.format(mAP, best_acc,
                global_step, best_global_step, optimizer.param_groups[0]['lr']))
                    model.train()
                    
                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()


def main():
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
