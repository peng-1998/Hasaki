import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate


class ColorMapType:
    COLORMAP_JET = 0
    COLORMAP_HSV = 1
    COLORMAP_RAINBOW = 3


class ColorMap():

    @torch.no_grad()
    def jet():
        r = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.00588235294117645, 0.02156862745098032, 0.03725490196078418, 0.05294117647058827, 0.06862745098039214, 0.084313725490196, 0.1000000000000001, 0.115686274509804, 0.1313725490196078, 0.1470588235294117, 0.1627450980392156, 0.1784313725490196, 0.1941176470588235, 0.2098039215686274,
            0.2254901960784315, 0.2411764705882353, 0.2568627450980392, 0.2725490196078431, 0.2882352941176469, 0.303921568627451, 0.3196078431372549, 0.3352941176470587, 0.3509803921568628, 0.3666666666666667, 0.3823529411764706, 0.3980392156862744, 0.4137254901960783, 0.4294117647058824,
            0.4450980392156862, 0.4607843137254901, 0.4764705882352942, 0.4921568627450981, 0.5078431372549019, 0.5235294117647058, 0.5392156862745097, 0.5549019607843135, 0.5705882352941174, 0.5862745098039217, 0.6019607843137256, 0.6176470588235294, 0.6333333333333333, 0.6490196078431372,
            0.664705882352941, 0.6803921568627449, 0.6960784313725492, 0.7117647058823531, 0.7274509803921569, 0.7431372549019608, 0.7588235294117647, 0.7745098039215685, 0.7901960784313724, 0.8058823529411763, 0.8215686274509801, 0.8372549019607844, 0.8529411764705883, 0.8686274509803922,
            0.884313725490196, 0.8999999999999999, 0.9156862745098038, 0.9313725490196076, 0.947058823529412, 0.9627450980392158, 0.9784313725490197, 0.9941176470588236, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9862745098039216, 0.9705882352941178, 0.9549019607843139, 0.93921568627451, 0.9235294117647062, 0.9078431372549018, 0.892156862745098, 0.8764705882352941, 0.8607843137254902, 0.8450980392156864, 0.8294117647058825,
            0.8137254901960786, 0.7980392156862743, 0.7823529411764705, 0.7666666666666666, 0.7509803921568627, 0.7352941176470589, 0.719607843137255, 0.7039215686274511, 0.6882352941176473, 0.6725490196078434, 0.6568627450980391, 0.6411764705882352, 0.6254901960784314, 0.6098039215686275,
            0.5941176470588236, 0.5784313725490198, 0.5627450980392159, 0.5470588235294116, 0.5313725490196077, 0.5156862745098039, 0.5
        ]
        g = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001960784313725483, 0.01764705882352935, 0.03333333333333333, 0.0490196078431373, 0.06470588235294117, 0.08039215686274503, 0.09607843137254901, 0.111764705882353, 0.1274509803921569,
            0.1431372549019607, 0.1588235294117647, 0.1745098039215687, 0.1901960784313725, 0.2058823529411764, 0.2215686274509804, 0.2372549019607844, 0.2529411764705882, 0.2686274509803921, 0.2843137254901961, 0.3, 0.3156862745098039, 0.3313725490196078, 0.3470588235294118, 0.3627450980392157,
            0.3784313725490196, 0.3941176470588235, 0.4098039215686274, 0.4254901960784314, 0.4411764705882353, 0.4568627450980391, 0.4725490196078431, 0.4882352941176471, 0.503921568627451, 0.5196078431372548, 0.5352941176470587, 0.5509803921568628, 0.5666666666666667, 0.5823529411764705,
            0.5980392156862746, 0.6137254901960785, 0.6294117647058823, 0.6450980392156862, 0.6607843137254901, 0.6764705882352942, 0.692156862745098, 0.7078431372549019, 0.723529411764706, 0.7392156862745098, 0.7549019607843137, 0.7705882352941176, 0.7862745098039214, 0.8019607843137255,
            0.8176470588235294, 0.8333333333333333, 0.8490196078431373, 0.8647058823529412, 0.8803921568627451, 0.8960784313725489, 0.9117647058823528, 0.9274509803921569, 0.9431372549019608, 0.9588235294117646, 0.9745098039215687, 0.9901960784313726, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9901960784313726, 0.9745098039215687, 0.9588235294117649, 0.943137254901961, 0.9274509803921571, 0.9117647058823528, 0.8960784313725489,
            0.8803921568627451, 0.8647058823529412, 0.8490196078431373, 0.8333333333333335, 0.8176470588235296, 0.8019607843137253, 0.7862745098039214, 0.7705882352941176, 0.7549019607843137, 0.7392156862745098, 0.723529411764706, 0.7078431372549021, 0.6921568627450982, 0.6764705882352944,
            0.6607843137254901, 0.6450980392156862, 0.6294117647058823, 0.6137254901960785, 0.5980392156862746, 0.5823529411764707, 0.5666666666666669, 0.5509803921568626, 0.5352941176470587, 0.5196078431372548, 0.503921568627451, 0.4882352941176471, 0.4725490196078432, 0.4568627450980394,
            0.4411764705882355, 0.4254901960784316, 0.4098039215686273, 0.3941176470588235, 0.3784313725490196, 0.3627450980392157, 0.3470588235294119, 0.331372549019608, 0.3156862745098041, 0.2999999999999998, 0.284313725490196, 0.2686274509803921, 0.2529411764705882, 0.2372549019607844,
            0.2215686274509805, 0.2058823529411766, 0.1901960784313728, 0.1745098039215689, 0.1588235294117646, 0.1431372549019607, 0.1274509803921569, 0.111764705882353, 0.09607843137254912, 0.08039215686274526, 0.06470588235294139, 0.04901960784313708, 0.03333333333333321, 0.01764705882352935,
            0.001960784313725483, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        b = [
            0.5, 0.5156862745098039, 0.5313725490196078, 0.5470588235294118, 0.5627450980392157, 0.5784313725490196, 0.5941176470588235, 0.6098039215686275, 0.6254901960784314, 0.6411764705882352, 0.6568627450980392, 0.6725490196078432, 0.6882352941176471, 0.7039215686274509, 0.7196078431372549,
            0.7352941176470589, 0.7509803921568627, 0.7666666666666666, 0.7823529411764706, 0.7980392156862746, 0.8137254901960784, 0.8294117647058823, 0.8450980392156863, 0.8607843137254902, 0.8764705882352941, 0.892156862745098, 0.907843137254902, 0.9235294117647059, 0.9392156862745098,
            0.9549019607843137, 0.9705882352941176, 0.9862745098039216, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9941176470588236,
            0.9784313725490197, 0.9627450980392158, 0.9470588235294117, 0.9313725490196079, 0.915686274509804, 0.8999999999999999, 0.884313725490196, 0.8686274509803922, 0.8529411764705883, 0.8372549019607844, 0.8215686274509804, 0.8058823529411765, 0.7901960784313726, 0.7745098039215685,
            0.7588235294117647, 0.7431372549019608, 0.7274509803921569, 0.7117647058823531, 0.696078431372549, 0.6803921568627451, 0.6647058823529413, 0.6490196078431372, 0.6333333333333333, 0.6176470588235294, 0.6019607843137256, 0.5862745098039217, 0.5705882352941176, 0.5549019607843138,
            0.5392156862745099, 0.5235294117647058, 0.5078431372549019, 0.4921568627450981, 0.4764705882352942, 0.4607843137254903, 0.4450980392156865, 0.4294117647058826, 0.4137254901960783, 0.3980392156862744, 0.3823529411764706, 0.3666666666666667, 0.3509803921568628, 0.335294117647059,
            0.3196078431372551, 0.3039215686274508, 0.2882352941176469, 0.2725490196078431, 0.2568627450980392, 0.2411764705882353, 0.2254901960784315, 0.2098039215686276, 0.1941176470588237, 0.1784313725490199, 0.1627450980392156, 0.1470588235294117, 0.1313725490196078, 0.115686274509804,
            0.1000000000000001, 0.08431372549019622, 0.06862745098039236, 0.05294117647058805, 0.03725490196078418, 0.02156862745098032, 0.00588235294117645, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        return torch.tensor([r, g, b]).T

    @torch.no_grad()
    def hsv():
        r = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9523809523809526, 0.8571428571428568, 0.7619047619047614, 0.6666666666666665, 0.5714285714285716, 0.4761904761904763, 0.3809523809523805, 0.2857142857142856, 0.1904761904761907, 0.0952380952380949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0.09523809523809557, 0.1904761904761905, 0.2857142857142854, 0.3809523809523809, 0.4761904761904765, 0.5714285714285714, 0.6666666666666663, 0.7619047619047619, 0.8571428571428574, 0.9523809523809523, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        g = [
            0, 0.09523809523809523, 0.1904761904761905, 0.2857142857142857, 0.3809523809523809, 0.4761904761904762, 0.5714285714285714, 0.6666666666666666, 0.7619047619047619, 0.8571428571428571, 0.9523809523809523, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9523809523809526,
            0.8571428571428577, 0.7619047619047619, 0.6666666666666665, 0.5714285714285716, 0.4761904761904767, 0.3809523809523814, 0.2857142857142856, 0.1904761904761907, 0.09523809523809579, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        b = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09523809523809523, 0.1904761904761905, 0.2857142857142857, 0.3809523809523809, 0.4761904761904762, 0.5714285714285714, 0.6666666666666666, 0.7619047619047619, 0.8571428571428571, 0.9523809523809523, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9523809523809526, 0.8571428571428577, 0.7619047619047614, 0.6666666666666665, 0.5714285714285716, 0.4761904761904767, 0.3809523809523805, 0.2857142857142856, 0.1904761904761907, 0.09523809523809579, 0
        ]
        return torch.tensor([r, g, b]).T

    @torch.no_grad()
    def rainbow():
        r = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9365079365079367, 0.8571428571428572, 0.7777777777777777, 0.6984126984126986, 0.6190476190476191, 0.53968253968254, 0.4603174603174605, 0.3809523809523814, 0.3015873015873018, 0.2222222222222223,
            0.1428571428571432, 0.06349206349206415, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03174603174603208, 0.08465608465608465, 0.1375661375661377, 0.1904761904761907, 0.2433862433862437, 0.2962962962962963, 0.3492063492063493, 0.4021164021164023, 0.4550264550264553, 0.5079365079365079,
            0.5608465608465609, 0.6137566137566139, 0.666666666666667
        ]
        g = [
            0, 0.03968253968253968, 0.07936507936507936, 0.119047619047619, 0.1587301587301587, 0.1984126984126984, 0.2380952380952381, 0.2777777777777778, 0.3174603174603174, 0.3571428571428571, 0.3968253968253968, 0.4365079365079365, 0.4761904761904762, 0.5158730158730158, 0.5555555555555556,
            0.5952380952380952, 0.6349206349206349, 0.6746031746031745, 0.7142857142857142, 0.753968253968254, 0.7936507936507936, 0.8333333333333333, 0.873015873015873, 0.9126984126984127, 0.9523809523809523, 0.992063492063492, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9841269841269842,
            0.9047619047619047, 0.8253968253968256, 0.7460317460317465, 0.666666666666667, 0.587301587301587, 0.5079365079365079, 0.4285714285714288, 0.3492063492063493, 0.2698412698412698, 0.1904761904761907, 0.1111111111111116, 0.03174603174603208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        b = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01587301587301582, 0.09523809523809534, 0.1746031746031744, 0.2539682539682535, 0.333333333333333, 0.412698412698413, 0.4920634920634921, 0.5714285714285712,
            0.6507936507936507, 0.7301587301587302, 0.8095238095238093, 0.8888888888888884, 0.9682539682539679, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
        return torch.tensor([r, g, b]).T

    colormapgetter = [jet, hsv, rainbow]

    def __init__(self, colormaptype: int = ColorMapType.COLORMAP_JET):
        self.colormap = self.colormapgetter[colormaptype]()

    @torch.no_grad()
    def get_map(self, length: int):
        return interpolate(self.colormap, size=[length, 3], mode='linear')


class IndexMapDyeing(Module):

    def __init__(self, colormap: Tensor | ColorMap, max_index: int, use_black_for_0: bool = True) -> None:
        '''
            Args:
                colormap:List of RGB color with shape (N,3) and type torch.float or a ColorMap instance
                max_index:The max value in your index-map 
                use_black_for_0:It will set black for map where index is 0 when it is True
        '''
        super().__init__()
        assert isinstance(colormap, Tensor | ColorMap)
        if isinstance(colormap, ColorMap):
            colormap = colormap.get_map(max_index + (0 if use_black_for_0 else 1))
            if use_black_for_0:
                colormap = torch.cat([torch.zeros(1, 3), colormap])
        assert colormap.shape.__len__() == 2 and colormap.shape[1] == 3
        self.colormap = colormap

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        assert input.shape.__len__() >= 2
        dims_order = [_ for _ in range(input.shape.__len__())]
        dims_order.insert(-3, 3)
        return self.colormap.to(input.device)[input].permute(*dims_order)


class BinaryMapContrast(Module):

    def __init__(self, predchannel: int = 0, lablechannel: int = 1) -> None:
        '''
        Args:
            predchannel:The index of channel in RGB image,the value should in [0,1,2]
            lablechannel:The index of channel in RGB image and should be different with predchannel
        '''
        assert predchannel in [0, 1, 2] and lablechannel in [0, 1, 2] and predchannel != lablechannel
        super().__init__()
        self.predchannel = predchannel
        self.lablechannel = lablechannel

    @torch.no_grad()
    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        assert pred.shape == label.shape and pred.shape.__len__() >= 2
        pred, label = pred.float(), label.float()
        image = torch.zeros(*pred.shape[:-2], 3, *pred.shape[-2:])
        image[..., self.predchannel, :, :] = pred
        image[..., self.lablechannel, :, :] = label
        return image
