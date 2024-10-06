# idea: select model you use in training and the trainer (the warper for training process)

import logging
import sys

sys.path.append('../../')

import torch
import torchvision.models as models
from torchvision.models.resnet import resnet34, resnet50
from typing import Optional
from torchvision.transforms import Resize
import torch.nn as nn
import torch.nn.functional as F
from utils.trainer_cls import ModelTrainerCLS

try:
    from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b3
except:
    logging.warning("efficientnet_b0,b3 fails to import, plz update your torch and torchvision")
try:
    from torchvision.models import mobilenet_v3_large
except:
    logging.warning("mobilenet_v3_large fails to import, plz update your torch and torchvision")

try:
    from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
except:
    logging.warning("vit fails to import, plz update your torch and torchvision")


def partially_load_state_dict(model, state_dict):
    # from https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113, thanks to chenyuntc Yun Chen
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        try:
            param = param.data
            own_state[name].copy_(param)
        except:
            print(f"unmatch: {name}")
            continue


from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
def vit_forword(self, x_ori):
    x = Resize((224, 224))(x_ori)
    # Reshape and permute the input tensor
    processed_x = self._process_input(x)
    bs = processed_x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(bs, -1, -1)     # [bs,1,768]
    processed_x = torch.cat([batch_class_token, processed_x], dim=1)
    mid_x = self.encoder(processed_x)     # [bs,197,768]
    # Classifier "token" as used by standard language architectures
    mid_x = mid_x[:, 0]     # [bs,768]
    out = self.heads(mid_x)   # [bs,10]
    return out, mid_x

def use_mid_nwdforward(self, mid_output):
    # mid_output: [bs,768]
    # dim = nn.Conv2d(512, 3, kernel_size=1).to('cuda:0')
    # print(f"dim.weight: {dim.weight.shape}, dim.bias: {dim.bias.shape}")
    out = self.dim(mid_output)
    # dim = nn.Conv2d(512,3,kernel_size=1)
    out = F.adaptive_avg_pool2d(out, 56)
    # out = F.conv2d(out,(128,3,1,1))
    # out = dim(out)
    # out = out.view(out.size(0), -1)
    # out = self.linear(out)
    return out

# class vit_b_16_wrap(vit_b_16):
#     # 子类继承基类
#     def __init__(self, ):
#         super().__init__(name)  # 调用基类的构造方法
#         net = vit_b_16(
#             pretrained=True,
#             **{k: v for k, v in kwargs.items() if k != "pretrained"}
#         )
#         net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
#         net = torch.nn.Sequential(
#             Resize((224, 224)),
#             net,
#         )
#
#     def get_mid_output(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         return out
#
#     def forward2(self, mid_output):
#         out = self.avgpool(mid_output)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
#     def get_mid_forward(self, x):
#         mid_output = self.get_mid_output(x)
#         out = self.forward2(mid_output)
#         return out, mid_output


# trainer is cls
def generate_cls_model(
        model_name: str,
        num_classes: int = 10,
        image_size: int = 32,
        **kwargs,
):
    '''
    # idea: aggregation block for selection of classifcation models
    :param model_name:
    :param num_classes:
    :return:
    '''

    logging.debug("image_size ONLY apply for vit!!!\nIf you use vit make sure you set the image size!")

    if model_name == 'resnet18':
        from torchvision.models.resnet import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'preactresnet18':
        logging.debug('Make sure you want PreActResNet18, which is NOT resnet18.')
        from models.preact_resnet import PreActResNet18
        if kwargs.get("pretrained", False):
            logging.warning("PreActResNet18 pretrained on cifar10, NOT ImageNet!")
            raise "debug"
            net_from_cifar10 = PreActResNet18()  # num_classes = num_classes)
            net_from_cifar10.load_state_dict(
                torch.load("../resource/trojannn/clean_preactresnet18.pt", map_location="cpu"
                           )['model_state_dict']
            )
            net = PreActResNet18(num_classes=num_classes)
            partially_load_state_dict(net, net_from_cifar10.state_dict())
        else:
            net = PreActResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        net = resnet34(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet50':
        net = resnet50(num_classes=num_classes, **kwargs)
    elif model_name == 'alexnet':
        net = models.alexnet(num_classes=num_classes, **kwargs)
    elif model_name == "vgg11":
        net = models.vgg11(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg16':
        net = models.vgg16(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19':
        net = models.vgg19(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        if kwargs.get("pretrained", False):
            net_from_imagenet = models.vgg19_bn(pretrained=True)  # num_classes = num_classes)
            net = models.vgg19_bn(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k != "pretrained"})
            partially_load_state_dict(net, net_from_imagenet.state_dict())
        else:
            net = models.vgg19_bn(num_classes=num_classes, **kwargs)
    elif model_name == 'squeezenet1_0':
        net = models.squeezenet1_0(num_classes=num_classes, **kwargs)
    elif model_name == 'densenet161':
        net = models.densenet161(num_classes=num_classes, **kwargs)
    elif model_name == 'inception_v3':
        net = models.inception_v3(num_classes=num_classes, **kwargs)
    elif model_name == 'googlenet':
        net = models.googlenet(num_classes=num_classes, **kwargs)
    elif model_name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0(num_classes=num_classes, **kwargs)
    elif model_name == 'mobilenet_v2':
        net = models.mobilenet_v2(num_classes=num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = models.mobilenet_v3_large(num_classes=num_classes, **kwargs)
    elif model_name == 'resnext50_32x4d':
        net = models.resnext50_32x4d(num_classes=num_classes, **kwargs)
    elif model_name == 'wide_resnet50_2':
        net = models.wide_resnet50_2(num_classes=num_classes, **kwargs)
    elif model_name == 'mnasnet1_0':
        net = models.mnasnet1_0(num_classes=num_classes, **kwargs)
    elif model_name == 'efficientnet_b0':
        net = efficientnet_b0(num_classes=num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = efficientnet_b3(num_classes=num_classes, **kwargs)
    elif model_name.startswith("vit"):
        logging.debug("All vit model use the default pretrain and resize to match the input shape!")

        if model_name == 'vit_b_16':
            net = vit_b_16(
                pretrained=True,
                **{k: v for k, v in kwargs.items() if k != "pretrained"}
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )

            # for vit model
            net.vit_forword = vit_forword
            net.use_mid_nwdforward = use_mid_nwdforward

        # elif model_name == 'vit_b_32':
        #     net = vit_b_32(
        #         pretrained=True,
        #         **{k: v for k, v in kwargs.items() if k != "pretrained"}
        #     )
        #     net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
        #     net = torch.nn.Sequential(
        #         Resize((224, 224)),
        #         net,
        #     )
        # elif model_name == 'vit_l_16':
        #     net = vit_l_16(
        #         pretrained=True,
        #         **{k: v for k, v in kwargs.items() if k != "pretrained"}
        #     )
        #     net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
        #     net = torch.nn.Sequential(
        #         Resize((224, 224)),
        #         net,
        #     )
        # elif model_name == 'vit_l_32':
        #     net = vit_l_32(
        #         pretrained=True,
        #         **{k: v for k, v in kwargs.items() if k != "pretrained"}
        #     )
        #     net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
        #     net = torch.nn.Sequential(
        #         Resize((224, 224)),
        #         net,
        #     )
    elif model_name == 'densenet121':
        net = models.densenet121(num_classes=num_classes, **kwargs)
    elif model_name == 'resnext29':
        from models.resnext import ResNeXt29_2x64d
        net = ResNeXt29_2x64d(num_classes=num_classes)
    elif model_name == 'senet18':
        from models.senet import SENet18
        net = SENet18(num_classes=num_classes)
    elif model_name == "convnext_tiny":
        logging.debug("All convnext model use the default pretrain!")
        from torchvision.models import convnext_tiny
        net_from_imagenet = convnext_tiny(pretrained=True, )  # num_classes = num_classes)
        net = convnext_tiny(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k != "pretrained"})
        partially_load_state_dict(net, net_from_imagenet.state_dict())
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net


def generate_cls_trainer(
        model,
        attack_name: Optional[str] = None,
        amp: bool = False,
):
    '''
    # idea: The warpper of model, which use to receive training settings.
        You can add more options for more complicated backdoor attacks.

    :param model:
    :param attack_name:
    :return:
    '''

    trainer = ModelTrainerCLS(
        model=model,
        amp=amp,
    )

    return trainer
