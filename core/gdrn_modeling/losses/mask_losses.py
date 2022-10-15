from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from einops import rearrange


def ex_loss_logits(logits, target):
    """Modified from: https://github.com/PengtaoJiang/OAA-
    PyTorch/blob/master/scripts/train_iam.py.

    Paper: http://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf.
    """
    assert logits.size() == target.size()
    scalar = torch.tensor([0]).float().cuda()
    pos = torch.gt(target, 0)
    neg = torch.eq(target, 0)
    pos_loss = -target[pos] * torch.log(torch.sigmoid(logits[pos]))
    neg_loss = -(torch.log(torch.exp(-(torch.max(logits[neg], scalar.expand_as(logits[neg])))) + 1e-8)) + torch.log(
        1 + torch.exp(-(torch.abs(logits[neg])))
    )

    loss = 0.0
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    if num_pos > 0:
        loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
    if num_neg > 0:
        loss += 1.0 / num_neg.float() * torch.sum(neg_loss)

    return loss


def ex_loss_probs(probs, target):
    """Modified from: https://github.com/PengtaoJiang/OAA-
    PyTorch/blob/master/scripts/train_iam.py.

    Paper: http://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf.
    """
    assert probs.size() == target.size()
    # scalar = torch.tensor([0]).float().cuda()
    pos = torch.gt(target, 0)
    neg = torch.eq(target, 0)
    probs = probs.clamp(min=1e-7, max=1 - 1e-7)
    pos_loss = -target[pos] * torch.log(probs[pos])
    neg_loss = -(torch.log(1 - probs[neg] + 1e-8))

    loss = 0.0
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    if num_pos > 0:
        loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
    if num_neg > 0:
        loss += 1.0 / num_neg.float() * torch.sum(neg_loss)

    return loss


def weighted_ex_loss_probs(probs, target, weight=None):
    """Modified from: https://github.com/PengtaoJiang/OAA-
    PyTorch/blob/master/scripts/train_iam.py.

    Paper: http://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf.
    """
    assert probs.size() == target.size()
    pos = torch.gt(target, 0)
    neg = torch.eq(target, 0)
    probs = probs.clamp(min=1e-7, max=1 - 1e-7)
    if weight is not None:
        pos_loss = -target[pos] * torch.log(probs[pos]) * weight[pos]
        neg_loss = -(torch.log(1 - probs[neg])) * weight[neg]
    else:
        pos_loss = -target[pos] * torch.log(probs[pos])
        neg_loss = -(torch.log(1 - probs[neg]))
    if torch.isnan(pos_loss).any():
        # print('pos_loss', pos_loss)
        print(
            "pos_loss nan",
            target.min(),
            target.max(),
            probs.min(),
            probs.max(),
        )
    if torch.isnan(neg_loss).any():
        # print('neg_loss', neg_loss)
        print(
            "neg_loss nan",
            target.min(),
            target.max(),
            probs.min(),
            probs.max(),
        )

    loss = 0.0
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    if num_pos > 0:
        loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
    if num_neg > 0:
        loss += 1.0 / num_neg.float() * torch.sum(neg_loss)

    return loss


def jaccard_loss_with_logits(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        logits: a tensor of shape [B, C, H, W].
            Corresponds to the raw output or logits of the model.
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)

        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()

        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


def jaccard_loss(probs, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        probs: a tensor of shape [B, C, H, W].
            Corresponds to the sigmoid/softmax of raw output or logits of the model.
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = probs.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)

        pos_prob = probs
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        probas = probs
    true_1_hot = true_1_hot.type(probs.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_score = (intersection / (union + eps)).mean()
    return 1 - jacc_score


def tversky_loss(probs, true, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].

        loss = 1 - |P*G| / (|P*G| + alpha*|P\G| + beta*|G\P| + eps)
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        probs: a tensor of shape [B, C, H, W]. Corresponds to
            softmax/sigmoid of the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = probs.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = probs  # torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        probas = probs  # F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(probs.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_score = (num / (denom + eps)).mean()
    return 1 - tversky_score


def focal_tversky_loss(probs, true, alpha=0.5, beta=0.7, gamma=0.75, eps=1e-7):
    """Computes the focal Tversky loss [1].

        loss = 1 - |P*G| / (|P*G| + alpha*|P\G| + beta*|G\P| + eps)
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        probs: a tensor of shape [B, C, H, W]. Corresponds to
            softmax/sigmoid of the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = probs.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = probs  # torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = rearrange(true_1_hot, "b h w c -> b c h w").float()
        probas = probs  # F.softmax(probas, dim=1)
    true_1_hot = true_1_hot.type(probs.type())
    # dims = (0,) + tuple(range(2, true.ndimension()))
    dims = tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_score = (num / (denom + eps)).mean()
    # gamma = 1/gamma < 1
    return torch.pow(1 - tversky_score, gamma)


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class JaccardLoss(_Loss):
    """
    Implementation of Jaccard loss for image segmentation task.
    Reference: https://github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/losses/jaccard.py

    It supports binary, multi-class and multi-label cases.
    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss=False,
        from_logits=True,
        smooth=0,
        eps=1e-7,
    ):
        """
        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation; By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor, weight=None) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        weight: Nx1xHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.softmax(dim=1)
            else:
                y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if weight is None:
            weight = torch.ones_like(y_true)
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            weight = weight.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            weight = weight.view(bs, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = rearrange(y_true, "N HW C -> N C HW")

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            self.smooth,
            self.eps,
            dims=dims,
            weight=weight,
        )

        if self.log_loss:
            loss = -(torch.log(scores))
        else:
            loss = 1 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = (y_true.sum(dims) > 0).float()
        loss = loss * mask

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


def soft_jaccard_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth=0.0,
    eps=1e-7,
    dims=None,
    weight=None,
) -> torch.Tensor:
    """
    Args:
        y_pred: (N, NC, *)
        y_true: (N, NC, *)
        smooth:
        eps:
        dims: dims to be summed
        weight: element-wise weight (should be the same as y_true)
    Returns:
        scalar
    """
    assert y_pred.size() == y_true.size()
    if weight is None:
        weight = 1.0

    if dims is not None:
        intersection = torch.sum(y_pred * y_true * weight, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true * weight)
        cardinality = torch.sum(y_pred + y_true)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth + eps)
    return jaccard_score


def weighted_soft_dice_loss(probs, labels, weights):
    """
    Args:
        probs: [B, 1, H, W]
        labels: [B, 1,H, W]
        weights: [B, 1, H, W]
    """
    num = labels.size(0)
    w = weights.view(num, -1)
    w2 = w * w
    m1 = probs.view(num, -1)
    m2 = labels.view(num, -1)
    intersection = m1 * m2
    smooth = 1.0
    score = 2.0 * ((w2 * intersection).sum(1) + smooth) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + smooth)
    loss = 1 - score.sum() / num
    return loss


def soft_dice_loss(probs, labels, smooth=0.0, eps=1e-7, reduction="mean"):
    """
    Args:
        probs: [B, 1, H, W]
        labels: [B, 1,H, W]
        eps: in SOLOv2, eps=0.002
    """
    num = labels.size(0)
    m1 = probs.view(num, -1)
    m2 = labels.view(num, -1)
    intersection = m1 * m2
    # smooth = 1.
    score = 2.0 * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth + eps)
    if reduction == "mean":
        loss = 1 - score.sum() / num
    elif reduction == "sum":
        loss = (1 - score).sum()
    else:
        loss = 1 - score
    return loss


def soft_dice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth=0,
    eps=1e-7,
    dims=None,
    weight=None,
) -> torch.Tensor:
    """
    Args:
        y_pred: (N, NC, *)
        y_true: (N, NC, *)
        smooth:
        eps:
    Returns:
        scalar
    """
    assert y_pred.size() == y_true.size()
    if weight is None:
        weight = 1.0
    if dims is not None:
        intersection = torch.sum(y_pred * y_true * weight, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true * weight)
        cardinality = torch.sum(y_pred + y_true)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth + eps)
    return dice_score


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

    raise ValueError("Unsupported input type" + str(type(x)))
