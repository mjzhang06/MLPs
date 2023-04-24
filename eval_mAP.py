# validation of parameters @mjz

import os
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP_eval(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        # print("scores:",scores)
        targets = targs[:, k]
        # print("targets:",targets)
        
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), 100 * ap


def AP_partial(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    cnt_class_with_no_neg = 0
    cnt_class_with_no_pos = 0
    cnt_class_with_no_labels = 0

    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]

        # Filter out samples without label
        idx = (targets != -1)
        scores = scores[idx]
        targets = targets[idx]
        if len(targets) == 0:
            cnt_class_with_no_labels += 1
            ap[k] = -1
            continue
        elif sum(targets) == 0:
            cnt_class_with_no_pos += 1
            ap[k] = -1
            continue
        if sum(targets == 0) == 0:
            cnt_class_with_no_neg += 1
            ap[k] = -1
            continue

        # compute average precision
        ap[k] = average_precision(scores, targets)

    idx_valid_classes = np.where(ap != -1)[0]
    ap_valid = ap[idx_valid_classes]
    map = 100 * np.mean(ap_valid)

    # Compute macro-map
    targs_macro_valid = targs[:, idx_valid_classes].copy()
    targs_macro_valid[targs_macro_valid <= 0] = 0  # set partial labels as negative
    n_per_class = targs_macro_valid.sum(0)  # get number of targets for each class
    n_total = np.sum(n_per_class)
    map_macro = 100 * np.sum(ap_valid * n_per_class / n_total)

    return ap, map, map_macro #,cnt_class_with_no_neg, cnt_class_with_no_pos, cnt_class_with_no_labels, 


def mAP_partial(targs, preds):
    """ mean Average precision for partial annotated validatiion set"""

    if np.size(preds) == 0:
        return 0
    results = AP_partial(targs, preds)
    mAP = results[1]
    return mAP

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


#======= AP, mAP计算 ======
def precision_score(gt, pred):
    """
    Params:
        gt:     {ndarray(N)} `0` or `1`
        pred:   {ndarray(N)} `0` or `1`
    Returns：
        p:      {float}
    """
    # print('in pr')
    # print(pred)
    # print(gt)
    index = pred == 1.  # 预测为1的index
    # print(index)
    # print(index.sum())
    _gt = gt[index]  # 预测为1的index对应的真实标签
    # print(_gt)
    tp = _gt[_gt == 1.].shape[0]  # 预测为1 且 真实标签为1 的 数量 （TP）
    tp = float(tp)
    # print(tp)
    pp = _gt.shape[0]  # 预测为1的数量
    pp = float(pp)
    # print(pp)
    p = tp / pp if pp != 0 else 0
    # print(p)
    return p


def recall_score(gt, pred):
    """
    Params:
        gt:     {ndarray(N)} `0` or `1`
        pred:   {ndarray(N)} `0` or `1`
    Returns：
        r:      {float}
    """
    # print('in re')
    # print(pred)
    # print(gt)
    index = gt == 1.    # 真实标签为1的index
    # print(index)
    _pred = pred[index] # 真实标签为1的index 对应的 预测标签
    # print(_pred)
    tp = _pred[_pred == 1.].shape[0]  # 真实标签为1的index 对应的 预测标签为1 的数量
    tp = float(tp)
    # print(tp)
    gp = _pred.shape[0] # 真实标签为1的数量
    gp = float(gp)
    # print(gp)
    r = tp / gp if gp != 0 else 0
    # print(r)
    return r

def _average_precision(rec, prec):
    """
    calculate average precision
    Params:
    ----------
    rec : numpy.array
        cumulated recall
    prec : numpy.array
        cumulated precision
    Returns:
    ----------
    ap as float
    """
    # print('ap')
    if rec is None or prec is None:
        return np.nan

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i])
    return ap


def score2label(pred_prob, t):
    return (pred_prob >= t).astype('float64')


def cal_ML_AP_mAP(pred_prob, gt):
    '''
    :param pred_prob: (N,cls_num) N个样本的cls_num个类的置信度
    :param gt: (N, cls_num) N个样本cls_num个类的ground-truth (0 or 1)
    :return: mAP, AP
    '''
    # print(pred_prob.shape)
    # print(gt.shape)
    cls_num = pred_prob.shape[1]
    AP = np.zeros(shape=(cls_num,))
    AP_sum = 0.
    for cls_i in range(cls_num):
        # print("cls_i: ",cls_i)
        # 计算第cls_i个类的AP
        _pred_prob = pred_prob[:, cls_i]  # 第cls_i个类所有样本的预测置信度
        _gt = gt[:, cls_i]  # 第cls_i个类所有样本的真实标签

        thresh = list(np.sort(_pred_prob))[::-1]
        _p, _r = [], []
        for t in thresh:
            _label = score2label(_pred_prob, t)  # 置信度超过阈值的样本, 标记为预测为1
            n = _pred_prob[_pred_prob == t].shape[0]  # 预测置信度等于阈值的样本数
            for i in range(n):
                _p += [precision_score(_gt, _label)]
                _r += [recall_score(_gt, _label)]
        # print('===p&r===')
        # print(_p, _r)
        _p = np.array(_p)
        _r = np.array(_r)
        _ap = _average_precision(_r, _p)
        # print(_ap)
        AP[cls_i] = _ap
        AP_sum += _ap
        # print(AP_sum)
    mAP = AP_sum / float(cls_num)
    return mAP, AP


def calculate_mAP_CCC(labels, preds):
    no_examples = labels.shape[0]
    no_classes = labels.shape[1]

    ap_scores = np.empty((no_classes), dtype=np.float)
    for ind_class in range(no_classes):
        ground_truth = labels[:, ind_class]
        out = preds[:, ind_class]

        sorted_inds = np.argsort(out)[::-1] # in descending order
        tp = ground_truth[sorted_inds]
        fp = 1 - ground_truth[sorted_inds]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / np.sum(ground_truth)
        prec = tp / (fp + tp)

        rec = np.insert(rec, 0, 0)
        rec = np.append(rec, 1)
        prec = np.insert(prec, 0, 0)
        prec = np.append(prec, 0)

        for ind in range(no_examples, -1, -1):
            prec[ind] = max(prec[ind], prec[ind + 1])

        inds = np.where(rec[1:] != rec[:-1])[0] + 1
        ap_scores[ind_class] = np.sum((rec[inds] - rec[inds - 1]) * prec[inds])

    return ap_scores, 100 * np.mean(ap_scores)

