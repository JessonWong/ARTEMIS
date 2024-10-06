import copy

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.models.util import box_ops
from utils.models.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                                    accuracy, get_world_size, interpolate,
                                    is_dist_avail_and_initialized)
from utils.function import normal, normal_style
from utils.function import calc_mean_std
from utils.models.ViT_helper import DropPath, to_2tuple, trunc_normal_

# 探究不同vgg编码层对蒸馏效果的影响: 改变这里面的处理层数
use_layers = 5
assert use_layers in [2,3,4,5]  # L1(12),L2(123),L3(1234),L4(12345)

device = "cuda:0"

# 根据输入的img得到patch
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)      # img尺寸: (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # patch尺寸: (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches  # patch数量

        # in_chans: img的通道数(RGB图像=3)
        # embed_dim: 嵌入维数
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# vgg网络结构的定义
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class StyTrans(nn.Module):  # trainnew.py中调用的network就是这个。所以要是想改vgg网络的层数，就改这里面的处理层
    """ This is the style transform transformer module """

    def __init__(self, encoder, decoder, PatchEmbed, transformer, args):
        # 参数:
        # encoder: vgg网络
        # decoder: StyTR.decoder网络
        # PatchEmbed: StyTR.PatchEmbed()
        # transformer: transformer.Transformer()

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False     # vgg单纯用于编码，全部不进行训练更新

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.decode = decoder
        self.embedding = PatchEmbed

    def encode_with_intermediate(self, input):  # encoder网络:vgg
        results = [input]
        for i in range(use_layers):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)      # 报错
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    # 根据特征图计算相似性loss
    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """

        # content_input = copy.deepcopy(samples_c)     # style (1,3,32,32)
        # style_input = copy.deepcopy(samples_s)     # style (1,3,32,32)
        samples_c = samples_c.cpu().detach().clone().to(device).requires_grad_(False)
        samples_s = samples_s.cpu().detach().clone().to(device).requires_grad_(False)
        content_input = samples_c   # img (bs,3,32,32)
        style_input =  samples_s     # style (1,3,32,32)
        assert samples_c.requires_grad==False
        assert samples_s.requires_grad==False
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(
                samples_c)  # support different-sized images padding is used for mask [tensor, mask]
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

            # ### features used to calcate loss
        # encoder的层数从2~5，返回的results列表的元素也从2~5
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        ### Linear projection
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask, content, pos_c, pos_s)   # bug
        Ics = self.decode(hs)
        Ics = Ics.detach()
        Ics.requires_grad_(False)

        # print(f"Ics.requires_grad : {Ics.requires_grad}")

        Ics_feats = self.encode_with_intermediate(Ics)
        # print(f"Ics_feats[-1].shape: {Ics_feats[-1].shape}, requires_grad={Ics_feats[-1].requires_grad}")
        # print(f"Ics_feats[-2].shape: {Ics_feats[-2].shape}, requires_grad={Ics_feats[-2].requires_grad}")
        # print(f"content_feats[-1].shape: {content_feats[-1].shape}, requires_grad={content_feats[-1].requires_grad}")
        # print(f"content_feats[-2].shape: {content_feats[-2].shape}, requires_grad={content_feats[-2].requires_grad}")
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + self.calc_content_loss(
            normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, use_layers):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))

        # Identity losses lambda 1
        loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)

        # Identity losses lambda 2
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0],
                                                                                                       style_feats[0])
        for i in range(1, use_layers):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(
                Iss_feats[i], style_feats[i])
        # Please select and comment out one of the following two sentences
        return Ics, loss_c, loss_s, loss_lambda1, loss_lambda2  # train
        # return Ics    #test
