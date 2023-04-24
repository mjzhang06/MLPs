# multi-label loss function
# config, @mjz
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn as nn, Tensor
import os
import pandas as pd
import numpy as np

# myBCELoss, BCEFocalLoss
class myBCELoss(nn.Module):
    def __init__(self, eps=1e-12, reduction='mean'):
        super(myBCELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, label):
        '''
        :param pred: sigmoid后的值
        :param label: 标签. 多标签分类的标签. 每个类为0/1
        :return: (batchsize, classes) 每个样本的每个类别上的loss.
        '''
        loss = -(torch.log(pred + self.eps) * label + torch.log(1. - pred + self.eps) * (1. - label))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict) # sigmoide获取概率
        pt = predict
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class myBFLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(myBFLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, targets):
        # predict = torch.sigmoid(predict) # sigmoide获取概率
        # loss = - (1 - pt) ** self.gamma * targets * torch.log(pt) - pt ** self.gamma * (1 - targets) * torch.log(1 - pt)
        # loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        
        # Calculating predicted probability
        xs_pos = predict
        xs_neg = 1 - predict
        eps = 1e-5

        # Focal for postive loss
        pt = (1 - xs_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # BCE loss calculation
        los_pos = targets * torch.log(xs_pos + eps)
        los_neg = (1 - targets) * torch.log(xs_neg + eps)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss. 

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'mean') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self, tau: float = 0.6, change_epoch: int = 1,
                 margin: float = 1.0, gamma: float = 2.0,
                 reduction: str = 'mean') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)
        
        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x
        # print(x_sigmoid)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1) # 最小数值和最大数值指定返回值的范围

        # Basic CE calculation, BCE Loss
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)       # 新的节点都是不可求导的
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.mean()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class myBCELoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=0, clip=0, eps=1e-12):
        super(myBCELoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        '''
        :param pred: sigmoid后的值
        :param label: 标签. 多标签分类的标签. 每个类为0/1
        :return: (batchsize, classes) 每个样本的每个类别上的loss.
        '''
        loss = -(torch.log(pred + self.eps) * label + torch.log(1. - pred + self.eps) * (1. - label))

        return loss  # (batchsize, class)


# ================== 2022.11.07 ==================
# =========== Generalized Dice Loss ==============
class wDice_loss(nn.Module):
    def __init__(self, reduction = 'mean', wight = 1):
        super(wDice_loss, self).__init__()
        self.reduction = reduction
        self.wight = wight

    def forward(self, logit, target):
        if not (target.size() == logit.size()):
            raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

        # preds = torch.sigmoid(logit)
        preds = logit
        preds_bg = 1 - preds  # bg = background
        preds = torch.cat([preds, preds_bg], dim=1)

        target_bg = 1 - target
        target = torch.cat([target, target_bg], dim=1)

        sp_dims = list(range(2, logit.dim()))
        weight = 1 / (1 + torch.sum(target, dim=sp_dims) ** 2)
        if self.wight == 1:
            generalized_dice = 2 * torch.sum(torch.sum(preds * target, dim=sp_dims), dim=-1) \
                / torch.sum(torch.sum(preds ** 2 + target ** 2, dim=sp_dims), dim=-1)
        else:
            generalized_dice = 2 * torch.sum(weight * torch.sum(preds * target, dim=sp_dims), dim=-1) \
                / torch.sum(weight * torch.sum(preds ** 2 + target ** 2, dim=sp_dims), dim=-1)

        loss = 1 - generalized_dice

        return loss.mean()


# ========== Asymmetric Similarity Loss ====================
class ASL_Loss(nn.Module):
    def __init__(self, beta=2, reduction='mean'):
        super(ASL_Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, logit, target):
        if not (target.size() == logit.size()):
            raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

        # preds = torch.sigmoid(logit)
        preds = logit

        sum_dims = list(range(1, logit.dim()))

        f_beta = (1 + self.beta ** 2) * torch.sum(preds * target, dim=sum_dims) \
                / ((1 + self.beta ** 2) * torch.sum(preds * target, dim=sum_dims) +
                    self.beta ** 2 * torch.sum((1 - preds) * target, dim=sum_dims) +
                    torch.sum(preds * (1 - target), dim=sum_dims))

        loss = 1 - f_beta

        return loss.mean()

        
## ========== 非平衡Loss ==========
class PartialSelectiveLoss(nn.Module):

    def __init__(self, args):
        super(PartialSelectiveLoss, self).__init__()
        self.args = args
        self.clip = args.clip
        self.gamma_pos = args.gamma_pos
        self.gamma_neg = args.gamma_neg
        self.gamma_unann = args.gamma_unann
        self.alpha_pos = args.alpha_pos
        self.alpha_neg = args.alpha_neg
        self.alpha_unann = args.alpha_unann

        self.targets_weights = None

        # if args.prior_path is not None:
        #     print("Prior file was found in given path.")
        #     df = pd.read_csv(args.prior_path)
        #     self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
        #     print("Prior file was loaded successfully. ")

    def forward(self, logits, targets):

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        prior_classes = None
        if hasattr(self, "prior_classes"):
            prior_classes = torch.tensor(list(self.prior_classes.values())).cuda()

        targets_weights = self.targets_weights
        targets_weights, xs_neg = edit_targets_parital_labels(self.args, targets, targets_weights, xs_neg,
                                                              prior_classes=prior_classes)

        # Loss calculation
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(xs_pos, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(xs_neg, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(xs_neg, min=1e-8))

        # BCE_loss = BCE_pos + BCE_neg + BCE_unann
        BCE_loss = BCE_pos + BCE_neg

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets_pos - xs_neg * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        return -BCE_loss.sum()


def edit_targets_parital_labels(args, targets, targets_weights, xs_neg, prior_classes=None):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if args.partial_loss_mode is None:
        targets_weights = 1.0

    elif args.partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets_weights = 1.0

    elif args.partial_loss_mode == 'ignore':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'ignore_normalize_classes':
        # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
        alpha_norm, beta_norm = 1, 1
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        n_annotated = 1 + torch.sum(targets != -1, axis=1)    # Add 1 to avoid dividing by zero

        g_norm = alpha_norm * (1 / n_annotated) + beta_norm
        n_classes = targets_weights.shape[1]
        targets_weights *= g_norm.repeat([n_classes, 1]).T
        targets_weights[targets == -1] = 0

    elif args.partial_loss_mode == 'selective':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_k = args.likelihood_topk * targets_weights.shape[0]

        xs_neg_prob = xs_neg
        if prior_classes is not None:
            if args.prior_threshold:
                idx_ignore = torch.where(prior_classes > args.prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

        negative_backprop_fun_jit(targets, xs_neg_prob, targets_weights, num_top_k)

        # set all unsure targets as negative
        # targets[targets == -1] = 0

    return targets_weights, xs_neg


def negative_backprop_fun_jit(targets: Tensor, xs_neg_prob: Tensor, targets_weights: Tensor, num_top_k: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_prob_flatten = xs_neg_prob.flatten()
        ind_class_sort = torch.argsort(xs_neg_prob_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_k]]] = 0


class ComputePrior:
    def __init__(self, classes):
        self.classes = classes
        n_classes = len(self.classes)
        self.sum_pred_train = torch.zeros(n_classes).cuda()
        self.sum_pred_val = torch.zeros(n_classes).cuda()
        self.cnt_samples_train,  self.cnt_samples_val = .0, .0
        self.avg_pred_train, self.avg_pred_val = None, None
        self.path_dest = "./outputs"
        self.path_local = "/class_prior/"

    def update(self, logits, training=True):
        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            if training:
                self.sum_pred_train += torch.sum(preds, axis=0)
                self.cnt_samples_train += preds.shape[0]
                self.avg_pred_train = self.sum_pred_train / self.cnt_samples_train

            else:
                self.sum_pred_val += torch.sum(preds, axis=0)
                self.cnt_samples_val += preds.shape[0]
                self.avg_pred_val = self.sum_pred_val / self.cnt_samples_val

    def save_prior(self):

        print('Prior (train), first 5 classes: {}'.format(self.avg_pred_train[:5]))

        # Save data frames as csv files
        if not os.path.exists(self.path_dest):
            os.makedirs(self.path_dest)

        df_train = pd.DataFrame({"Classes": list(self.classes.values()),
                                 "avg_pred": self.avg_pred_train.cpu()})
        df_train.to_csv(path_or_buf=os.path.join(self.path_dest, "train_avg_preds.csv"),
                        sep=',', header=True, index=False, encoding='utf-8')

        if self.avg_pred_val is not None:
            df_val = pd.DataFrame({"Classes": list(self.classes.values()),
                                   "avg_pred": self.avg_pred_val.cpu()})
            df_val.to_csv(path_or_buf=os.path.join(self.path_dest, "val_avg_preds.csv"),
                          sep=',', header=True, index=False, encoding='utf-8')

    def get_top_freq_classes(self):
        n_top = 10
        top_idx = torch.argsort(-self.avg_pred_train.cpu())[:n_top]
        top_classes = np.array(list(self.classes.values()))[top_idx]
        print('Prior (train), first {} classes: {}'.format(n_top, top_classes))



class GHME_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        '''
        bins: split to n bins
        alpha: hyper-parameter
        '''
        super(GHME_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target

    def forward(self, x, target):
        # g = torch.abs(self._custom_loss_grad(x, target)).detach()
        g = torch.abs(x.detach() - target)

        bin_idx = self._g2bin(g)
        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))
        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()
        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=1)    # 将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        beta = N / gd
        # print(beta)

        # BCE loss calculation
        los_pos = target * torch.log(x)
        los_neg = (1 - target) * torch.log(1-x)

        loss = -(los_pos + los_neg)
        loss *= beta[bin_idx].to(self.device)
        return loss.mean()

        # return self._custom_loss(x, target, beta[bin_idx]).to(self.device)


class GHMC_Loss(GHME_Loss):
    '''
        GHM_Loss for classification
    '''

    def __init__(self, bins=10, alpha=0.5):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHME_Loss):
    '''
        GHM_Loss for regression
    '''

    def __init__(self, bins=10, alpha=0.5, mu=0.02):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


# ============= 可运行的GHMC代码，分类任务 =========
class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5, is_split_batch=False):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None
        self.is_split_batch = is_split_batch
        self.is_evaluation = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_evaluation(self, is_evaluation):
        """
        评估时可以(也可以不管)将is_evaluation设为True,训练时设为False，这样就类似于直接计算CEL
        :param is_evaluation: bool
        :return:
        """
        self.is_evaluation = is_evaluation

    def _g2bin(self, g, bin):
        return torch.floor(g * (bin - 0.0001)).long()

    def use_alpha(self, bin_count):
        if (self._alpha != 0):
            if (self.is_evaluation):
                if (self._last_bin_count == None):
                    self._last_bin_count = bin_count
                else:
                    bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
                    self._last_bin_count = bin_count
        return bin_count

    def forward(self, x, target):
        """
        :param x: torch.Tensor,[B,C,*]
        :param target: torch.Tensor,[B,*]
        :return: loss
        """
        g = torch.abs(x.detach() - target)
        weight = torch.zeros((x.size(0), x.size(-1)))
        if self.is_split_batch:
            #是否对每个batch分开统计梯度，我实验时发现分开统计loss会更容易收敛，可能因为模型中用了batch normalization？
            N = x.size(0) * x.size(1)
            bin = (int)(N // self._bins)
            bin_idx = self._g2bin(g, bin)
            # bin_idx = torch.clamp(bin_idx, max=bin - 1)
            bin_count = torch.zeros((x.size(0), bin))
            for i in range(x.size(0)):
                bin_count[i] = torch.from_numpy(np.bincount(torch.flatten(bin_idx[i].cpu()), minlength=bin))
                bin_count[i] *= (bin_count[i] > 0).sum().item()

            gd = self.use_alpha(bin_count)
            gd = torch.clamp(gd, min=1)
            beta = N * 1.0 / gd
            for i in range(x.size(0)):
                weight[i] = beta[i][bin_idx[i]]
        else:
            N = x.size(0) * x.size(1)
            bin = (int)(N // self._bins)
            bin_idx = self._g2bin(g, bin)
            # bin_idx = torch.clamp(bin_idx, max=bin - 1) # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。
            bin_count = torch.from_numpy(np.bincount(torch.flatten(bin_idx.cpu()), minlength=bin))
            bin_count *= (bin_count > 0).sum().item()

            gd = self.use_alpha(bin_count)
            gd = torch.clamp(gd, min=1) # min = 0.0001
            beta = N * 1.0 / gd
            weight = beta[bin_idx]
        # print(weight)

        # BCE loss calculation
        los_pos = target * torch.log(x)
        los_neg = (1 - target) * torch.log(1-x)

        loss = -(los_pos + los_neg)
        # print(loss)
        loss = loss * weight.to(self.device)

        return loss.mean()
        