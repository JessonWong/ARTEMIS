import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.grl import WarmStartGradientReverseLayer
import utils.models.transformer as transformer
import utils.models.StyTR as StyTR
import matplotlib.pyplot as plt
import os

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"

class params(object):
    def __init__(self):
        current_pwd = os.getcwd()
        self.current_pwd = current_pwd
        print(f"【in nwd.py】os.getcwd(): {current_pwd}")
        self.style_dir = os.path.join(current_pwd, '../utils/input/style/S_image1.jpg')
        self.vgg = os.path.join(current_pwd, '../utils/experiments/vgg_normalised.pth')
        self.decoder_path = os.path.join(current_pwd, '/./utils/experiments/decoder_iter_160000.pth')
        self.Trans_path = os.path.join(current_pwd, '../utils/experiments/transformer_iter_160000.pth')
        self.embedding_path = os.path.join(current_pwd, '../utils/experiments/embedding_iter_160000.pth')
        if "/home/xuemeng" in os.getcwd():     # 在chariot使用
            self.style_dir = os.path.join(current_pwd, 'utils/input/style/S_image1.jpg')
            self.vgg = os.path.join(current_pwd, 'utils/experiments/vgg_normalised.pth')
            self.decoder_path = os.path.join(current_pwd, 'utils/experiments/decoder_iter_160000.pth')
            self.Trans_path = os.path.join(current_pwd, 'utils/experiments/transformer_iter_160000.pth')
            self.embedding_path = os.path.join(current_pwd, 'utils/experiments/embedding_iter_160000.pth')


p = params()
vgg = StyTR.vgg
vgg.load_state_dict(torch.load(p.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
decoder = StyTR.decoder
embedding = StyTR.PatchEmbed().to(device)
embedding = embedding.to(device)
Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, p)
network.to(device)


class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)    # 用了暖启动技术，在反传时用于梯度反转的函数
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        y_s = y_s.to(device)
        y_t = y_t.to(device)
        out, loss_c, loss_s, l_identity1, l_identity2 = network(y_s, y_t)  # network：transformer
        return loss_s

    def forward(self, f: torch.Tensor) -> torch.Tensor: # 输入: f:(image, 风格迁移后的image)
        f_grl = self.grl(f)     # 得到梯度
        # f_grl=f_grl.unsqueeze(0)
        y_s, y_t = f_grl.chunk(2, dim=0)        # 原样本的梯度 和 迁移样本的梯度
        loss = self.n_discrepancy(y_s, y_t)     # 输入transformer得到loss
        # if loss > 1e3:
        #     loss = 0
        # print(f"type: {type(loss)}, loss: {loss}")
        return loss
