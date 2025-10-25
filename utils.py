import os
import sys
import json
import pickle
import random
import numpy as np

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # accumulated loss
    accu_num = torch.zeros(1).to(device)   # number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = pred.squeeze(1)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)   # number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = pred.squeeze(1)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def train_one_epoch_reg(model, optimizer, data_loader, device, epoch, reg_loss):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # accumulated loss
    accu_num = torch.zeros(1).to(device)   # number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = pred.squeeze(1)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device)) + reg_loss(model)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_reg(model, data_loader, device, epoch, reg_loss):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)   # number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = pred.squeeze(1)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device)) + reg_loss(model)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch_mvckd_collaborated(model, alpha, T, optimizer, data_loader, device, epoch):
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    accu_loss = torch.zeros(1).to(device)  # accumulated loss
    accu_num = torch.zeros(1).to(device)   # number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, teacher_emds_1, teacher_emds_2, teacher_emds_3 = data
        sample_num += images.shape[0]

        # patch-level FM as student
        patch_pred = model(images.to(device), role='student')
        patch_pred = patch_pred.squeeze(1)
        pred_classes = torch.max(patch_pred, dim=1)[1]

        # wsi-level FM as teacher
        wsi_pred_1 = model(teacher_emds_1.to(device), role='teacher_1')
        wsi_pred_1 = wsi_pred_1.squeeze(1)

        wsi_pred_2 = model(teacher_emds_2.to(device), role='teacher_2')
        wsi_pred_2 = wsi_pred_2.squeeze(1)

        wsi_pred_3 = model(teacher_emds_3.to(device), role='teacher_3')
        wsi_pred_3 = wsi_pred_3.squeeze(1)

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        smoothed_patch_pred = torch.nn.functional.log_softmax(patch_pred / T, dim=1)
        smoothed_wsi_pred_1 = torch.nn.functional.softmax(wsi_pred_1 / T, dim=1)
        smoothed_wsi_pred_2 = torch.nn.functional.softmax(wsi_pred_2 / T, dim=1)
        smoothed_wsi_pred_3 = torch.nn.functional.softmax(wsi_pred_3 / T, dim=1)

        loss = alpha*ce_loss(patch_pred, labels.to(device)) + (1/3)*(1 - alpha)*(kl_loss(smoothed_patch_pred, smoothed_wsi_pred_1)+kl_loss(smoothed_patch_pred, smoothed_wsi_pred_2)+kl_loss(smoothed_patch_pred, smoothed_wsi_pred_3))*(T*T)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_mvckd_collaborated(model, alpha, T, data_loader, device, epoch):
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    model.eval()

    accu_num = torch.zeros(1).to(device)   # number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, teacher_emds_1, teacher_emds_2, teacher_emds_3 = data
        sample_num += images.shape[0]

        # patch-level FM as student
        patch_pred = model(images.to(device), role='student')
        patch_pred = patch_pred.squeeze(1)
        pred_classes = torch.max(patch_pred, dim=1)[1]

        # wsi-level FM as teacher
        wsi_pred_1 = model(teacher_emds_1.to(device), role='teacher_1')
        wsi_pred_1 = wsi_pred_1.squeeze(1)

        wsi_pred_2 = model(teacher_emds_2.to(device), role='teacher_2')
        wsi_pred_2 = wsi_pred_2.squeeze(1)

        wsi_pred_3 = model(teacher_emds_3.to(device), role='teacher_3')
        wsi_pred_3 = wsi_pred_3.squeeze(1)

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        smoothed_patch_pred = torch.nn.functional.log_softmax(patch_pred / T, dim=1)
        smoothed_wsi_pred_1 = torch.nn.functional.softmax(wsi_pred_1 / T, dim=1)
        smoothed_wsi_pred_2 = torch.nn.functional.softmax(wsi_pred_2 / T, dim=1)
        smoothed_wsi_pred_3 = torch.nn.functional.softmax(wsi_pred_3 / T, dim=1)
        loss = alpha * ce_loss(patch_pred, labels.to(device)) + (1 / 3) * (1 - alpha) * (kl_loss(smoothed_patch_pred, smoothed_wsi_pred_1) + kl_loss(smoothed_patch_pred,smoothed_wsi_pred_2) + kl_loss(smoothed_patch_pred, smoothed_wsi_pred_3)) * (T * T)

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)





