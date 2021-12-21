from typing import Optional
from torch import Tensor
import torch
import torch.nn.functional as F


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
        return dice_loss(F.softmax(pred, dim=1) if pred.min() < 0 or pred.max() > 1 else pred, label, weight=self.weight if self.weight is not None else weight, batch_dice=self.batch_dice, square=self.square, smooth=self.smooth)


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
        label = torch.one_hot(label, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *range(1, len(pred.shape) - 1)).contiguous()
    anb = weight * pred * label

    if square:
        aub = weight * ((pred**2) + (label**2))
    else:
        aub = weight * (pred + label)
    if batch_dice:
        anb = anb.sum()
        aub = aub.sum()
    else:
        anb = anb.sum(*range(1, len(anb.shape) - 1))
        aub = aub.sum(*range(1, len(anb.shape) - 1))
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
