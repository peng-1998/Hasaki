from typing import List, Tuple
import torch
from torch import Tensor
from scipy.ndimage import label as bmap_label
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F

def rand_error(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the rand error between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    
    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([0,0,1,1])
        >>> print(rand_error(input,target))
        >>> 0.0
    """

    assert isinstance(input, Tensor) and isinstance(target, Tensor) and input.shape == target.shape
    n = torch.prod(torch.tensor(input.shape)).item()
    m = (input == target).sum().item()
    return 2 * (n - m) * m / n / (n - 1)


def rand_score(input: Tensor, target: Tensor) -> float:
    """
    Return the rand score,which mean 1 - rand error.
    See:"~Hasaki.score.rand_error"
    """
    return 1 - rand_error(input, target)


def F1_score(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the F1 score between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    
    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.5
    """

    assert isinstance(input, Tensor) and isinstance(target, Tensor) and input.shape == target.shape
    pl   = input + target
    tp   = (pl == 2).sum().item()
    fpfn = (pl == 1).sum().item()
    return tp / (tp + 0.5 * fpfn + 1e-6)


def IoU(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the IoU between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    
    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.3333333333
    """

    assert isinstance(input, Tensor) and isinstance(target, Tensor) and input.shape == target.shape
    pl   = input + target
    tp   = (pl == 2).sum().item()
    fpfn = (pl == 1).sum().item()
    return tp / (tp + fpfn + 1e-6)


def Dice_score(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the Dice score between input and target.
    See ~Hasaki.score.F1
    """

    return F1_score(input, target)


def Jaccard_score(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the Jaccard Index between input and target.
    See ~Hasaki.score.IoU
    """

    return IoU(input, target)


def AJI_For_Binary_Map(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the Aggregated Jaccard Index between binary map input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    """

    assert input.shape == target.shape and len(input.shape) == 3
    score = 0
    for i in range(input.shape[0]):
        (input_objs, _)   = bmap_label(input[i].cpu().numpy())
        input_objs        = torch.tensor(input_objs).to(torch.long).to(input.device)
        (target_objs, _)  = bmap_label(target[i].cpu().numpy())
        target_objs       = torch.tensor(target_objs).to(torch.long).to(input.device)
        score           += _aji(input_objs, target_objs)
    return score / input.shape[0]


def AJI(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the Aggregated Jaccard Index between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    """
    
    assert input.shape == target.shape and len(input.shape) == 3
    score = 0
    input  = input.to(torch.long)
    target = target.to(torch.long)
    for i in range(input.shape[0]):
        score += _aji(input[i], target[i])
    return score / input.shape[0]


def _aji(input: Tensor, target: Tensor) -> float:
    device              = input.device
    input_objs           = torch._C._nn.one_hot(input, num_classes=input.max() + 1).permute(2, 0, 1)[1:].to(device)
    target_objs          = torch._C._nn.one_hot(target, num_classes=target.max() + 1).permute(2, 0, 1)[1:].to(device)
    input_objs           = input_objs.unsqueeze(1).to(torch.float32)
    target_objs          = target_objs.unsqueeze(1).to(torch.float32)
    if input_objs.shape[0] == 0:
        return 0
    ands                = F.conv2d(target_objs, input_objs).squeeze()
    ors                 = torch.prod(torch.tensor(input.shape)) - F.conv2d(1 - target_objs, 1 - input_objs).squeeze()
    ious                = ands / ors
    (_,index)           = ious.max(dim=-1)
    index_              = torch._C._nn.one_hot(index, input.max())
    C                   = (index_ * ands).sum()
    U                   = (index_ * ors).sum()
    index               = index.unique(sorted=False)
    U                  += (input != 0).sum() - torch.index_select(input_objs, dim=0, index=index).sum() + 1e-6
    return (C / U).item()


def accuracy(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the accuracy of predictions.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Element in background should be 0 and different integer in every instance.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.

    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.5
    """

    assert input.shape == target.shape
    return ((input == target).sum() / torch.prod(torch.tensor(input.shape))).item()


def precision(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the precision of predictions.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Element in background should be 0 and different integer in every instance.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.

    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.5
    """

    assert input.shape == target.shape
    return (((input + target) == 2).sum() / (input == 1).sum()).item()


def recall(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the recall rate of predictions.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Element in background should be 0 and different integer in every instance.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.

    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.5
    """

    assert input.shape == target.shape
    return (((input + target) == 2).sum() / (target == 1).sum()).item()


def specificity(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the specificity of predictions.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Element in background should be 0 and different integer in every instance.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.

    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.5
    """
    assert input.shape == target.shape
    return (((input + target) == 0).sum() / (target == 0).sum()).item()


def sensitivity(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the sensitivity of predictions.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Element in background should be 0 and different integer in every instance.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.

    Examples::
        >>> input  = torch.tensor([1,1,0,0])
        >>> target = torch.tensor([1,0,0,1])
        >>> print(F1_score(input,target))
        >>> 0.5
    """

    assert input.shape == target.shape
    return (((input + target) == 2).sum() / (target == 1).sum()).item()


def Hausdorff_Distance(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the Hausdorff istance between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    """

    distances = []
    for i in range(input.shape[0]):
        edge_input       = get_edge(input[i])
        edge_lable       = get_edge(target[i])
        dist_input       = get_dist_map(input[i])
        dist_lable       = get_dist_map(target[i])
        input_edge_dist  = edge_input * dist_lable
        target_edge_dist = edge_lable * dist_input
        max_input_dist   = input_edge_dist.max()
        max_target_dist  = target_edge_dist.max()
        distances.append(max_target_dist if max_target_dist > max_input_dist else max_input_dist)
    return torch.stack(distances).mean().item()


def Hausdorff_Distance_95(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the 95% Hausdorff istance between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    """
    distances = []
    for i in range(input.shape[0]):
        edge_input          = get_edge(input[i])
        edge_lable          = get_edge(target[i])
        dist_input          = get_dist_map(input[i])
        dist_lable          = get_dist_map(target[i])
        input_edge_dist     = edge_input * dist_lable
        target_edge_dist    = edge_lable * dist_input
        (max_input_dist,_)  = input_edge_dist.topk(k=int((edge_input.sum()/20).item()))
        (max_target_dist,_) = target_edge_dist.topk(k=int((edge_lable.sum()/20).item()))
        distances.append(max_target_dist[-1] if max_target_dist[-1] > max_input_dist[-1] else max_input_dist[-1])
    return torch.stack(distances).mean().item()


def get_dist_map(map: Tensor):
    assert len(map.shape) < 4
    np_map   = map.cpu().numpy()
    dist_map = distance_transform_edt(1 - np_map)
    return torch.Tensor(dist_map, device=map.device)


def get_edge(map: Tensor) -> Tensor:
    assert len(map.shape) < 4
    if len(map.shape) == 3:
        kernel = [
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
            [
                [0,1,0],
                [1,1,1],
                [0,1,0]
            ],
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ]
        ]
        kernel = torch.tensor(kernel).view(1, 1, 3, 3, 3).to(map.device)
        map    = map.unsqueeze(0).unsqueeze(0)
        map_   = 1.0*(F.conv3d(map,kernel,padding=1)==kernel.sum())
        edge   = map - map_
    else:
        kernel = [
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ]
        kernel = torch.tensor(kernel).view(1, 1, 3, 3).to(map.device)
        map    = map.unsqueeze(0).unsqueeze(0)
        map_   = 1.0*(F.conv2d(map,kernel,padding=1)==kernel.sum())
        edge   = map - map_
    return edge


def ASD(input: Tensor, target: Tensor) -> float:
    """
    This indicator computes the Average Surface Distance between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
    Return::
        Return a float between 0 and 1.
    """

    distances = []
    for i in range(input.shape[0]):
        edge_input       = get_edge(input[i])
        edge_lable       = get_edge(target[i])
        dist_input       = get_dist_map(input[i])
        dist_lable       = get_dist_map(target[i])
        input_edge_dist  = edge_input * dist_lable
        target_edge_dist = edge_lable * dist_input
        distances.append((input_edge_dist.sum() + target_edge_dist.sum()) / (edge_lable.sum() + edge_input.sum()))
    return torch.stack(distances).mean().item()


def Surface_overlap(input: Tensor, target: Tensor, eps:float = 1.415) -> Tuple[float, float]:
    """
    This indicator computes the surface overlap rate between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
        eps:Two pixels are considered to overlap if their Euclidean distance is less than eps.
    Return::
        Return two float distance input -> target and target -> input.
    """

    distances_i_t = []
    distances_t_i = []
    for i in range(input.shape[0]):
        edge_input       = get_edge(input[i])
        edge_target      = get_edge(target[i])
        dist_input       = get_dist_map(input[i])
        dist_target      = get_dist_map(target[i])
        input_edge_dist  = edge_input * dist_target
        target_edge_dist = edge_target * dist_input
        distances_i_t.append((input_edge_dist < eps).sum() / edge_input.sum())
        distances_t_i.append((target_edge_dist < eps).sum() / edge_target.sum())
    return torch.stack(distances_i_t).mean().item(),torch.stack(distances_t_i).mean().item()


def Surface_dice(input: Tensor, target: Tensor, eps:float = 1.415) -> float:
    """
    This indicator computes the surface dice between input and target.

    Args::
        input:The predictions of the segmentation task.
            It should have shape (N,W,H) for 2D or (N,D,W,H) for 3D.
            Each element in it should be 1 for target and 0 for other.
        target:The masks of the segmentation task.
            It should have same shape and format with input.
        eps:Two pixels are considered to overlap if their Euclidean distance is less than eps.
    Return::
        Return Return a float between 0 and 1.
    """

    dice = []
    for i in range(input.shape[0]):
        edge_input       = get_edge(input[i])
        edge_target      = get_edge(target[i])
        dist_input       = get_dist_map(input[i])
        dist_target      = get_dist_map(target[i])
        input_edge_dist  = edge_input * dist_target
        target_edge_dist = edge_target * dist_input
        dice.append(((input_edge_dist < eps).sum() + (target_edge_dist < eps).sum()) / (edge_input.sum() + edge_target.sum()))
    return torch.stack(dice).mean().item()