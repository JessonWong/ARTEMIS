'''
This file implements the defense method called finetuning (ft), which is a standard fine-tuning that uses clean data to finetune the model.

basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. ft defense:
        a. get some clean data
        b. retrain the backdoor model
    4. test the result and get ASR, ACC, RC 
'''

import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import yaml
import logging
import time
from defense.base import defense
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainerdefense import PureCleanModelTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

# 降低训练数据量：平均每个label的样本数 = 数据集总大小 * ratio / label数量
dataset_num_ratio = 0
test_batchsize = 256


class ft(defense):

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str, help='the location of attack result')
        parser.add_argument('--save_path', type=str, help='the location of saving defense result')

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=int)
        parser.add_argument('--dataset_num_ratio', type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/ft/config.yaml", help='the path of yaml')

        # set the parameter for the ft defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

    def set_result(self, args):
        attack_file = args.result_file
        save_path = args.save_path
        assert save_path is not None
        os.makedirs(save_path, exist_ok=True)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file)

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
        print("self.device: ", self.device)
        print("****************************************")

    def mitigation(self):
        self.set_devices()
        
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.load_state_dict(self.result['model'])

        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)

        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        self.set_trainer(model)
        criterion = argparser_criterion(args)

        # 只用一部分的数据集进行训练，不然每个epoch太慢了。
        def get_ramdomsplit_dataset(train_dataset, bs=256, ratio=0.2):
            ratio = min(ratio, 1)
            # 为每个类选择相同数量的样本
            class_indices = {}
            for i in range(len(train_dataset)):  # 把所有数据的idx根据label分好类
                items = train_dataset[i]
                label = items[1]
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)
            subset_indices = []
            totalnum = 0
            for indices_list in class_indices.values():
                totalnum += len(indices_list)
            label_num = len(class_indices)
            num_samples_per_label = round(totalnum * ratio / label_num)
            print(
                f" --- totalnum: {totalnum}, label_num: {label_num}, num_samples_per_label: {num_samples_per_label} --- ")
            for indices in class_indices.values():
                subset_indices.extend(indices[:num_samples_per_label])
            # 创建 SubsetRandomSampler 对象来读取子集数据集
            subset_sampler = SubsetRandomSampler(subset_indices)
            subset_dataloader = DataLoader(train_dataset, batch_size=bs,
                                           num_workers=self.args.num_workers, shuffle=False,
                                           pin_memory=args.pin_memory, sampler=subset_sampler)
            return subset_dataloader

        train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                   train=True)  # 数据增强
        data_train = self.result['clean_train']
        data_train.wrap_img_transform = train_tran
        # data_loader = torch.utils.data.DataLoader(data_train, batch_size=self.args.batch_size,
        #                                           num_workers=self.args.num_workers, shuffle=True,
        #                                           pin_memory=args.pin_memory)
        print(f"cutting train dataset to ratio {dataset_num_ratio * 100:.2f}%")
        data_loader = get_ramdomsplit_dataset(data_train, self.args.batch_size, dataset_num_ratio)
        trainloader = data_loader

        test_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_bd_testset = self.result['bd_test']    # 只有bd数据集,如果是all2one类型的攻击,则label只有1个。
        data_bd_testset.wrap_img_transform = test_tran
        # data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size,
        #                                              num_workers=self.args.num_workers, drop_last=False, shuffle=True,
        #                                              pin_memory=args.pin_memory)
        print(f"cutting mixtest dataset to ratio {dataset_num_ratio * 100:.2f}%")
        data_bd_loader = get_ramdomsplit_dataset(data_bd_testset, test_batchsize, dataset_num_ratio)    # for de
        # data_bd_loader = get_ramdomsplit_dataset(data_bd_testset, test_batchsize, dataset_num_ratio + 1)


        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        # data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=test_batchsize,
        #                                                 num_workers=self.args.num_workers, drop_last=False,
        #                                                 shuffle=True, pin_memory=args.pin_memory)
        print(f"cutting test dataset to ratio {dataset_num_ratio*100:.2f}%")
        data_clean_loader = get_ramdomsplit_dataset(data_clean_testset, test_batchsize, dataset_num_ratio)
        # data_clean_loader = get_ramdomsplit_dataset(data_clean_testset, test_batchsize, dataset_num_ratio + 1)


        # self.trainer.train_with_test_each_epoch(
        #     train_data = trainloader,
        #     test_data = data_clean_loader,
        #     adv_test_data = data_bd_loader,
        #     end_epoch_num = self.args.epochs,
        #     criterion = criterion,
        #     optimizer = optimizer,
        #     scheduler = scheduler,
        #     device = self.args.device,
        #     frequency_save = self.args.frequency_save,
        #     save_folder_path = self.args.checkpoint_save,
        #     save_prefix = 'defense',
        #     continue_training_path = None,
        # )

        self.trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            data_clean_loader,
            data_bd_loader,
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='ours',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self, args):
        self.set_result(args)
        self.set_logger()
        result = self.mitigation()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    ft.add_arguments(parser)
    args = parser.parse_args()
    ft_method = ft(args)
    dataset_num_ratio = args.dataset_num_ratio
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = ft_method.defense(args)  # args.result_file要求是: 攻击结果目录的相对或绝对路径

# param:
# --batch_size
# 16
# --epoch
# 25
# --num_workers
# 1
# --result_file
# ../../BackdoorBench-ori/record/Attack_badnet_gtsrb_record/attack_result.pt
# --yaml_path
# ../config/defense/ft/gtsrb.yaml
# --save_path
# ../record/defense/badnet_defense/badnet_defense_L3
