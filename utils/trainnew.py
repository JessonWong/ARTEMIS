import copy
import os
import torch
import torch.nn as nn
from collections import OrderedDict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"in trainnew : torch.cuda.is_available: {torch.cuda.is_available()}")
device = "cuda:0"


### bug: 为什么一个worker会有两次输出，一次有cuda，一次没cuda？？？
class styletrans(object):
    def __init__(self, Trans, args, StyTR, style):
        # 加载vgg网络
        self.vgg = StyTR.vgg    # vgg结构 在StyTR.py中定义
        self.vgg.load_state_dict(torch.load(args.vgg))  # load 预训练的vgg参数
        self.vgg = nn.Sequential(*list(self.vgg.children())[:44])

        self.decoder = StyTR.decoder
        self.Trans = Trans
        self.embedding = StyTR.PatchEmbed()

        self.decoder.eval()
        self.Trans.eval()
        self.vgg.eval()

        self.StyTR = StyTR
        self.ori_style = style.unsqueeze(0)  # (1,3,32,32)
        self.style = copy.deepcopy(self.ori_style)
        self.args = args

        new_state_dict = OrderedDict()
        state_dict = torch.load(self.args.decoder_path)
        for k, v in state_dict.items():
            # namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.decoder.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load(self.args.Trans_path)
        for k, v in state_dict.items():
            # namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.Trans.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load(self.args.embedding_path)
        for k, v in state_dict.items():
            # namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.embedding.load_state_dict(new_state_dict)

        self.network = self.StyTR.StyTrans(self.vgg, self.decoder, self.embedding, self.Trans, self.args)
        self.network.eval()

    def __call__(self, img):
        if len(img.shape) == 3:     # assert img shape = (bs,3,32,32)
            img = img.unsqueeze(0)
        bs = img.shape[0]
        # print(f"bs:{bs}")

        if img.device != device:
            img = img.to(device)
        if self.ori_style.device != device:
            self.ori_style = self.ori_style.to(device)  # assert style shape = (bs,3,32,32)
            self.style = self.style.to(device)
        if self.style.shape[0] != bs:  # 最后一个batch可能不足bs个
            self.style = self.ori_style.repeat(bs, 1, 1, 1)
        self.network.to("cuda:0")

        with torch.no_grad():
            output = self.network(img, self.style)
            output = output[0].squeeze(0)
        return output
