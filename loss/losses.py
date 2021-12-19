from typing import Optional
from torch import Tensor
import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self, batch_dice: bool = True, square: bool = False, smooth: float = 1) -> None:
        '''
        Args:
            batch_dice:It will calculate the average loss of instances when parameter is False.
            square:Whether or not you square the denominator.
            smooth:The smoothing term of the loss function.
        '''
        super().__init__()
        self.batch_dice = batch_dice
        self.square = square
        self.smooth = smooth

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        assert len(pred.shape) == 4 or len(pred.shape) == 5
        if pred.shape[1] == 2:
            return dice_loss(F.softmax(pred, dim=1)[:, 1] if pred.min() < 0 or pred.max() > 1 else pred[:, 1], label)
        return dice_loss(F.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred, label)


def dice_loss(pred: Tensor, label: Tensor, batch_dice: bool = False, square: bool = False, smooth: float = 1) -> Tensor:
    assert pred.shape == label.shape
    anb = pred * label

    if square:
        aub = (pred**2) + (label**2)
    else:
        aub = pred + label
    if batch_dice:
        anb = anb.sum()
        aub = aub.sum()
    else:
        anb = anb.sum(*range(1, len(anb.shape) - 1))
        aub = aub.sum(*range(1, len(anb.shape) - 1))
    return 1 - (2 * anb / aub).mean()


def cross_entropy_with_weight_map(pred: Tensor, label: Tensor, weight: Tensor) -> Tensor:
    assert len(pred.shape) == 4 or len(pred.shape) == 5
    if len(pred.shape) == len(weight.shape) + 1:
        weight = weight.unsqueeze(1)
    if len(pred.shape) == len(weight.shape) + 2:
        weight = weight.unsqueeze(0).unsqueeze(0)
    if len(pred.shape) == len(label.shape) + 1:
        assert (pred.shape[:1] + pred.shape[2:]) == label.shape
        label = F.one_hot(label, num_classes=pred.shape[1]).permute(0, len(pred.shape) - 1, *[i for i in range(1, len(pred.shape) - 1)])
    if pred.min() < 0 or pred.max() > 1:
        return F.binary_cross_entropy_with_logits(pred, label, weight)
    return F.binary_cross_entropy(pred, label, weight)


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
