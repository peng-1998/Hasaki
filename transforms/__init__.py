from typing import Any, Tuple, Union
from torch.nn import Module
from . import transforms2D,transforms3D


class Compose(Module):
    '''
    The transform list like it in torchvision.transforms.Unlike torchvision.transforms.Compose,this Compose can get multiple input. 
    Use it like composeobj(img) of composeobj(img1,img2,...) and will return img or tuple[img,img,...].
    '''

    def __init__(self, transforms: list) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, *args) -> Union[Any,Tuple[Any,...]]:
        for t in self.transforms:
            args = t(*args)
        return args[0] if len(args) == 1 else tuple(args)


class TransformPart(Module):
    '''
    Only transform i-th image in inputs for i in \'indexes\'. 
    For example: 
    obj = TransformOnlyFrist(CenterCrop((100,100)),indexes=[0,2])  
    img1,img2,img3=obj(img1,img2,img3) 
    Input images have size (3,200,200) and return images will be (3,100,100),(3,200,200),(3,100,100). 
    This class is use to make class in torchvision.transforms get multiple input.
    '''

    def __init__(self, transform, indexes: list) -> None:
        super().__init__()
        self.transform = transform
        self.indexes = indexes

    def forward(self, *args) -> list:
        return [self.transform(_) if i in self.indexes else _ for i, _ in enumerate(args)]


class TransformOnlyFrist(TransformPart):
    '''
    Only transform frist image in inputs. 
    For example: 
    obj = TransformOnlyFrist(CenterCrop((100,100)))  
    img1,img2,img3=obj(img1,img2,img3) 
    Input images have size (3,200,200) and return images will be (3,100,100),(3,200,200),(3,200,200). 
    This class is use to make class in torchvision.transforms get multiple input.
    '''

    def __init__(self, transform) -> None:
        super().__init__(transform, [0])


class TransformForAll(Module):
    '''
    Transform all images use \'transform\'.But don\'t use it with random transforms. 
    This class is use to make class in torchvision.transforms get multiple input.
    '''

    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform

    def forward(self, *args) -> list:
        return [self.transform(_) for _ in args]

