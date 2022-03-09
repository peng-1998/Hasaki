import torch
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class DiceLoss(torch.nn.Module):
    def __init__(self, weight: Tensor = None, batch_dice: bool = True, square: bool = False, smooth: float = 1) -> None:
        '''
        Args:
            batch_dice: It will calculate the average loss of instances when parameter is False.
            square    : Whether or not you square the denominator.
            smooth    : The smoothing term of the loss function.
        '''
        super().__init__()
        self.weight = weight
        self.batch_dice = batch_dice
        self.square = square
        self.smooth = smooth

    def forward(self, pred: Tensor, label: Tensor, weight: Tensor = 1) -> Tensor:
        assert len(pred.shape) == 4 or len(pred.shape) == 5
        return dice_loss(pred, label, weight=self.weight if self.weight is not None else weight, batch_dice=self.batch_dice, square=self.square, smooth=self.smooth)


class IoULoss(DiceLoss):
    def forward(self, pred: Tensor, label: Tensor, weight: Tensor = 1) -> Tensor:
        D = super().forward(pred, label, weight)
        return 2 * D / (1 + D)


class WeightCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__(size_average=size_average, ignore_index=ignore_index, reduce=False, label_smoothing=label_smoothing)
        self.weight_map = weight
        self.reduction_ = reduction

    def forward(self, pred, lable):
        res = super().forward(pred, lable)
        if self.weight_map is not None:
            res = self.weight_map * res
        if self.reduction_ == 'mean':
            return res.mean()
        return res.sum()


def dice_loss(pred: Tensor, label: Tensor, weight: Tensor = 1, batch_dice: bool = False, square: bool = False, smooth: float = 1) -> Tensor:
    '''
    Args:
        pred      : Probability map of prediction with shape (N,C,H,W) or (N,C,D,H,W).
        label     : Binary map with shape (N,H,W) or (N,D,H,W).
        weight    : Weight map with shape (N,1,H,W) or (1,1,H,W) for 2D and (N,1,D,H,W) or (1,1,D,H,W) for 3D. 
        batch_dice: It will calculate the average loss of instances when parameter is False.
        square    : Whether or not you square the denominator.
        smooth    : The smoothing term of the loss function.
    '''
    if len(pred.shape) == len(label.shape) + 1:
        label = F.one_hot(label, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *range(1, len(pred.shape) - 1)).contiguous()
    anb = weight * pred * label

    if square:
        aub = weight * ((pred**2) + (label**2))
    else:
        aub = weight * (pred + label)
    if batch_dice:
        anb = anb.sum()
        aub = aub.sum()
    else:
        anb = anb.sum(tuple([i for i in range(1, len(anb.shape) - 1)]))
        aub = aub.sum(tuple([i for i in range(1, len(anb.shape) - 1)]))
    return 1 - (2 * anb / aub).mean()


def focal_loss(pred: Tensor, label: Tensor, alpha: float = 0.25, gamma: float = 2) -> Tensor:
    return alpha * F.binary_cross_entropy(pred, label, (1 - pred)**gamma)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2) -> None:
        super().__init__()
        assert alpha > 0 and gamma >= 0
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        return focal_loss(pred, label, self.alpha, self.gamma)


class TopKLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, ignore_index: int = -100, k: int = 10) -> None:
        super().__init__(weight=weight, ignore_index=ignore_index, reduce=False)
        self.k = k

    def forward(self, pred: Tensor, label: Tensor):
        if label.shape[1] == 1:
            label = label[:, 1]
        loss = super().forward(pred, label)
        num = torch.prod(torch.tensor(loss.shape)).item()
        loss, _ = torch.topk(loss.view(-1), int(num * self.k / 100), sorted=False)
        return loss.mean()


class SensitivitySpecificityLoss(DiceLoss):
    def __init__(self, w: float = 0.5, batch_dice: bool = True, square: bool = False, smooth: float = 1) -> None:
        super().__init__(weight=None, batch_dice=batch_dice, square=square, smooth=smooth)
        self.w = w

    def forward(self, pred: Tensor, label: Tensor):
        if len(pred.shape) == len(label.shape) + 1:
            label = F.one_hot(label, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *range(1, len(pred.shape) - 1)).contiguous()
        pred_label2 = (label - pred)**2
        _label = 1 - label
        return self.w * super().forward(pred_label2, label) + (1 - self.w) * super().forward(pred_label2, _label)


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1, beta: float = 1, gamma: float = 1, batch_loss=False) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_loss = batch_loss

    def forward(self, pred: Tensor, label: Tensor):
        if len(pred.shape) == len(label.shape) + 1:
            label = F.one_hot(label, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *range(1, len(pred.shape) - 1)).contiguous()
        predlabel = pred * label
        pred_label = pred * (1 - label)
        _predlabel = (1 - pred) * label
        if self.batch_loss:
            predlabel = predlabel.sum(tuple([i for i in range(1, len(predlabel.shape) - 1)]))
            pred_label = pred_label.sum(tuple([i for i in range(1, len(pred_label.shape) - 1)]))
            _predlabel = _predlabel.sum(tuple([i for i in range(1, len(_predlabel.shape) - 1)]))
        else:
            predlabel = predlabel.sum()
            pred_label = pred_label.sum()
            _predlabel = _predlabel.sum()
        return (1 - (predlabel / (predlabel + self.alpha * pred_label + self.beta * _predlabel)).mean())**self.gamma


class GeneralizedDiceloss(DiceLoss):
    def __init__(self, weight: Tensor = None, batch_dice: bool = True, square: bool = False, smooth: float = 1) -> None:
        assert len(weight.shape) == 1
        super().__init__(weight=None, batch_dice=batch_dice, square=square, smooth=smooth)
        self.class_weight = weight

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        return super().forward(pred, label, weight=self.weight.view(self.weight.shape[0], *[1 for _ in range(len(pred.shape) - 2)]))


class AsymmetricSimilarityLoss(TverskyLoss):
    def __init__(self, beta: float = 1.5, batch_loss=False) -> None:
        super().__init__(alpha=1 / (1 + beta**2), beta=beta**2 / (1 + beta**2), gamma=1, batch_loss=batch_loss)


class PenaltyLoss(GeneralizedDiceloss):
    def __init__(self, weight: Tensor = None, k: float = 2.5, batch_dice: bool = True, square: bool = False, smooth: float = 1) -> None:
        super().__init__(weight=weight, batch_dice=batch_dice, square=square, smooth=smooth)
        self.k = k

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        lgd = super().forward(pred, label)
        return (self.k - 1) * lgd / (1 + self.k * lgd)


def hausdorff_distance_loss(pred: Tensor, label: Tensor) -> Tensor:
    pred_bmap = pred.argmax(dim=1)
    pred_bmap = F.one_hot(pred_bmap, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *range(1, len(pred.shape) - 1)).cpu().numpy()
    if len(pred.shape) == len(label.shape) + 1:
        label = F.one_hot(label, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *range(1, len(pred.shape) - 1))
        label_bmap = label.cpu().numpy()
    else:
        label_bmap = label.cpu().numpy()
    dist = pred.cpu().numpy()
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            dist[i, j, ...] = distance_transform_edt(pred_bmap[i, j, ...]) + distance_transform_edt(label_bmap[i, j, ...])
    return (torch.from_numpy(dist).to(pred.device) * (pred - label)).sum(dim=1).mean()
