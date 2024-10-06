# This script is for trainer. This is a warpper for training process.
import copy
import os
import torch
from PIL import Image
import sys, logging

sys.path.append('../')
import random
from pprint import pformat
from typing import *
import pandas as pd

from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import time
from copy import deepcopy

from utils.prefetch import PrefetchLoader, prefetch_transform
from utils.nwd import NuclearWassersteinDiscrepancy
from utils.trainnew import styletrans
import utils.models.transformer as transformer
import utils.models.StyTR as StyTR

from utils.util_1.accuracy import Accuracy, AverageAccuracy
from utils.util_1.share_steps import shared_configure_optimizers

alpha = 0.5  # 蒸馏超参
noise_sigma = 2   # 噪声域中噪声的大小

show_progress = False
use_img_style_transfer = False  # 是否使用图像域迁移
partial_transfer = True
transfer_img_num = 3  # 只对前三张进行图像迁移

if partial_transfer:
    logging.info(f" 只对前几张图片进行图像迁移! \n")
else:
    logging.info(f" 对所有图片都进行图像迁移! \n")


class params(object):
    def __init__(self):
        current_pwd = os.getcwd()
        self.current_pwd = current_pwd
        print(f"【in trainerdefense.py】 os.getcwd(): {current_pwd}")
        self.style_dir = os.path.join(current_pwd, '../utils/input/style/S_image1.jpg')
        self.vgg = os.path.join(current_pwd, '../utils/experiments/vgg_normalised.pth')
        self.decoder_path = os.path.join(current_pwd, '../utils/experiments/decoder_iter_160000.pth')
        self.Trans_path = os.path.join(current_pwd, '../utils/experiments/transformer_iter_160000.pth')
        self.embedding_path = os.path.join(current_pwd, '../utils/experiments/embedding_iter_160000.pth')
        if "/home/xuemeng" in os.getcwd():  # 在chariot使用
            self.style_dir = os.path.join(current_pwd, 'utils/input/style/S_image1.jpg')
            self.vgg = os.path.join(current_pwd, 'utils/experiments/vgg_normalised.pth')
            self.decoder_path = os.path.join(current_pwd, 'utils/experiments/decoder_iter_160000.pth')
            self.Trans_path = os.path.join(current_pwd, 'utils/experiments/transformer_iter_160000.pth')
            self.embedding_path = os.path.join(current_pwd, 'utils/experiments/embedding_iter_160000.pth')


p = params()
style_transforms = transforms.Compose([
    # transforms.Resize([64, 64]),    # tiny是64
    transforms.Resize([32, 32]),    # gtsrb,cifar是32
    transforms.ToTensor()
])
style = style_transforms(Image.open(p.style_dir).convert("RGB"))
device = torch.device('cuda')
trans = styletrans(transformer.Transformer(), p, StyTR, style)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class dataloader_generator:
    def __init__(self, **kwargs_init):
        self.kwargs_init = kwargs_init

    def __call__(self, *args, **kwargs_call):
        kwargs = deepcopy(self.kwargs_init)
        kwargs.update(kwargs_call)
        return DataLoader(
            *args,
            **kwargs
        )


def last_and_valid_max(col: pd.Series):
    '''
    find last not None value and max valid (not None or np.nan) value for each column
    :param col:
    :return:
    '''
    return pd.Series(
        index=[
            'last', 'valid_max', 'exist_nan_value'
        ],
        data=[
            col[~col.isna()].iloc[-1], pd.to_numeric(col, errors='coerce').max(), any(i == 'nan_value' for i in col)
        ])


class Metric_Aggregator(object):
    '''
    aggregate the metric to log automatically
    '''

    def __init__(self):
        self.history = []

    def __call__(self,
                 one_metric: dict):
        one_metric = {k: v for k, v in one_metric.items() if v is not None}  # drop pair with None as value
        one_metric = {
            k: (
                "nan_value" if v is np.nan or torch.tensor(v).isnan().item() else v  # turn nan to str('nan_value')
            ) for k, v in one_metric.items()
        }
        self.history.append(one_metric)
        logging.info(
            pformat(
                one_metric
            )
        )

    def to_dataframe(self):
        self.df = pd.DataFrame(self.history, dtype=object)
        logging.debug("return df with np.nan and None converted by str()")
        return self.df

    def summary(self):
        '''
        do summary for dataframe of record
        :return:
        eg.
            ,train_epoch_num,train_acc_clean
            last,100.0,96.68965148925781
            valid_max,100.0,96.70848846435547
            exist_nan_value,False,False

        '''
        if 'df' not in self.__dict__:
            logging.debug('No df found in Metric_Aggregator, generate now')
            self.to_dataframe()
        logging.debug("return df with np.nan and None converted by str()")
        return self.df.apply(last_and_valid_max)


class ModelTrainerCLS():
    def __init__(self, model, amp=False):
        self.model = model
        self.amp = amp

    def init_or_continue_train(self,
                               end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler,
                               device,
                               continue_training_path: Optional[str] = None,
                               only_load_model: bool = False,
                               ) -> None:
        '''
        config the training process, from 0 or continue previous.
        The requirement for saved file please refer to save_all_state_to_path
        :param train_data: train_data_loader, only if when you need of number of batch, you need to input it. Otherwise just skip.
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.

        '''

        model = self.model

        model.to(device, non_blocking=True)
        model.train()

        # train and update

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        if continue_training_path is not None:
            logging.info(f"No batch info will be used. Cannot continue from specific batch!")

            start_epoch, _ = self.load_from_path(continue_training_path, device, only_load_model)
            self.start_epochs, self.end_epochs = start_epoch, end_epoch_num
        else:
            self.start_epochs, self.end_epochs = 0, end_epoch_num
            # self.start_batch = 0

        logging.info(f'All setting done, train from epoch {self.start_epochs} to epoch {self.end_epochs}')

        logging.info(
            pformat(f"self.amp:{self.amp}," +
                    f"self.criterion:{self.criterion}," +
                    f"self.optimizer:{self.optimizer}," +
                    f"self.scheduler:{self.scheduler.state_dict() if self.scheduler is not None else None}," +
                    f"self.scaler:{self.scaler.state_dict() if self.scaler is not None else None})")
        )

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_all_state_to_path(self,
                               path: str,
                               epoch: Optional[int] = None,
                               batch: Optional[int] = None,
                               only_model_state_dict: bool = False) -> None:
        '''
        save all information needed to continue training, include 3 random state in random, numpy and torch
        :param path: where to save
        :param epoch: which epoch when save
        :param batch: which batch index when save
        :param only_model_state_dict: only save the model, drop all other information
        '''

        save_dict = {
            'epoch_num_when_save': epoch,
            'batch_num_when_save': batch,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'model_state_dict': self.get_model_params(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
            "scaler": self.scaler.state_dict(),
        } \
            if only_model_state_dict == False else self.get_model_params()

        torch.save(
            save_dict,
            path,
        )

    def load_from_path(self,
                       path: str,
                       device,
                       only_load_model: bool = False
                       ) -> [Optional[int], Optional[int]]:
        '''

        :param path:
        :param device: map model to which device
        :param only_load_model: only_load_model or not?
        '''

        self.model = self.model.to(device, non_blocking=True)

        load_dict = torch.load(
            path, map_location=device
        )

        logging.info(f"loading... keys:{load_dict.keys()}, only_load_model:{only_load_model}")

        attr_list = [
            'epoch_num_when_save',
            'batch_num_when_save',
            'random_state',
            'np_random_state',
            'torch_random_state',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'criterion_state_dict',
        ]

        if all([key_name in load_dict for key_name in attr_list]):
            # all required key can find in load dict
            # AND only_load_model == False
            if only_load_model == False:
                random.setstate(load_dict['random_state'])
                np.random.set_state(load_dict['np_random_state'])
                torch.random.set_rng_state(load_dict['torch_random_state'].cpu())  # since may map to cuda

                self.model.load_state_dict(
                    load_dict['model_state_dict']
                )
                self.optimizer.load_state_dict(
                    load_dict['optimizer_state_dict']
                )
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(
                        load_dict['scheduler_state_dict']
                    )
                self.criterion.load_state_dict(
                    load_dict['criterion_state_dict']
                )
                if 'scaler' in load_dict:
                    self.scaler.load_state_dict(
                        load_dict["scaler"]
                    )
                    logging.info(f'load scaler done. scaler={load_dict["scaler"]}')
                logging.info('all state load successful')
                return load_dict['epoch_num_when_save'], load_dict['batch_num_when_save']
            else:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None

        else:  # only state_dict

            if 'model_state_dict' in load_dict:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None
            else:
                self.model.load_state_dict(
                    load_dict,
                )
                logging.info('only model state_dict load')
                return None, None

    def test(self, test_data, device):
        model = self.model
        model.to(device, non_blocking=True)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
        }

        criterion = self.criterion.to(device, non_blocking=True)

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_data):
                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    # @resource_check
    def train_one_batch(self, x, labels, device):

        self.model.train()
        self.model.to(device, non_blocking=True)

        x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.amp):
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels.long())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        batch_loss = loss.item() * labels.size(0)

        return batch_loss

    def train_one_epoch(self, train_data, device):
        startTime = time.time()
        batch_loss = []
        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
            batch_loss.append(self.train_one_batch(x, labels, device))
        one_epoch_loss = sum(batch_loss)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # here since ReduceLROnPlateau need the train loss to decide next step setting.
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time.time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        return one_epoch_loss

    def train(self, train_data, end_epoch_num,
              criterion,
              optimizer,
              scheduler, device, frequency_save, save_folder_path,
              save_prefix,
              continue_training_path: Optional[str] = None,
              only_load_model: bool = False, ):
        '''

        simplest train algorithm with init function put inside.

        :param train_data: train_data_loader
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''

        self.init_or_continue_train(
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train, epoch_loss: {epoch_loss[-1]}')
            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")

    def train_with_test_each_epoch(self,
                                   train_data,
                                   test_data,
                                   bd_test_data,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):
        '''
        train with test on clean and backdoor dataloader for each epoch

        :param train_data: train_data_loader
        :param test_data: clean test data
        :param adv_test_data: backdoor poisoned test data (for ASR)
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            metrics = self.test(test_data, device)
            metric_info = {
                'epoch': epoch,
                'clean acc': metrics['test_correct'] / metrics['test_total'],
                'clean loss': metrics['test_loss'],
            }
            agg(metric_info)

            bd_metrics = self.test(bd_test_data, device)
            bd_metric_info = {
                'epoch': epoch,
                'ASR': bd_metrics['test_correct'] / bd_metrics['test_total'],
                'backdoor loss': bd_metrics['test_loss'],
            }
            agg(bd_metric_info)

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
            agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")

    def train_with_test_each_epoch_v2(self,
                                      train_data,
                                      test_dataloader_dict,
                                      end_epoch_num,
                                      criterion,
                                      optimizer,
                                      scheduler,
                                      device,
                                      frequency_save,
                                      save_folder_path,
                                      save_prefix,
                                      continue_training_path: Optional[str] = None,
                                      only_load_model: bool = False,
                                      ):
        '''
        v2 can feed many test_dataloader, so easier for test with multiple dataloader.

        only change the test data part, instead of predetermined 2 dataloader, you can input any number of dataloader to test
        with {
            test_name (will show in log): test dataloader
        }
        in log you will see acc and loss for each test dataloader

        :param test_dataloader_dict: { name : dataloader }

        :param train_data: train_data_loader
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            for dataloader_name, test_dataloader in test_dataloader_dict.items():
                metrics = self.test(test_dataloader, device)
                metric_info = {
                    'epoch': epoch,
                    f'{dataloader_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dataloader_name} loss': metrics['test_loss'],
                }
                agg(metric_info)

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
            agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")

    def train_with_test_each_epoch_v2_sp(self,
                                         batch_size,
                                         train_dataset,
                                         test_dataset_dict,
                                         end_epoch_num,
                                         criterion,
                                         optimizer,
                                         scheduler,
                                         device,
                                         frequency_save,
                                         save_folder_path,
                                         save_prefix,
                                         prefetch=False,
                                         continue_training_path: Optional[str] = None,
                                         only_load_model: bool = False,
                                         ):

        '''
        Nothing different, just be simplified to accept dataset instead.
        '''
        train_data = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            num_workers=8,
        )

        test_dataloader_dict = {
            name: DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=seed_worker,
                num_workers=8,
            )
            for name, test_dataset in test_dataset_dict.items()
        }

        if prefetch:
            raise SystemError("Due to technical issue, not implemented yet")

        self.train_with_test_each_epoch_v2(
            train_data,
            test_dataloader_dict,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            frequency_save,
            save_folder_path,
            save_prefix,
            continue_training_path,
            only_load_model,
        )


def all_acc(preds: torch.Tensor,
            labels: torch.Tensor, ):
    if len(preds) == 0 or len(labels) == 0:
        logging.warning("zero len array in func all_acc(), return None!")
        return None
    return preds.eq(labels).sum().item() / len(preds)


def class_wise_acc(
        preds: torch.Tensor,
        labels: torch.Tensor,
        selected_class: list,
):
    assert len(preds) == len(labels)
    acc = {class_idx: 0 for class_idx in selected_class}
    for c in acc.keys():
        acc[c] = preds.eq(c).sum().item() / len(preds)
    return acc


def given_dataloader_test(
        model,
        test_dataloader,
        criterion,
        non_blocking: bool = False,
        device="cpu",
        verbose: int = 0
):
    model.to(device, non_blocking=non_blocking)
    model.eval()
    metrics = {
        'test_correct': 0,
        'test_loss_sum_over_batch': 0,
        'test_total': 0,
    }
    criterion = criterion.to(device, non_blocking=non_blocking)

    if verbose == 1:
        batch_predict_list, batch_label_list = [], []

    batch_length = len(test_dataloader)
    print(f"in trainerdefense, batch_length: {batch_length}")
    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
            # print(f"bs: {x.shape[0]}")    # 和 train的bs一样。
            if show_progress:
                logging.info(f" test batch {batch_idx}/{batch_length}")
            x = x.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            test_bs = x.shape[0]
            test_trans_num = min(test_bs, transfer_img_num)

            if use_img_style_transfer:
                if partial_transfer:
                    # for i in range(transfer_img_num):
                    #     tar0[i] = trans(image[i])  # 只对前 transfer_img_num 张图片进行图像迁移【串行】
                    x[:transfer_img_num] = trans(x[:transfer_img_num])
                else:
                    x = trans(x)

            # 因为经历了风格迁移，所以acc和ra一开始会很低。
            pred = model(x)
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            if verbose == 1:
                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(target.detach().clone().cpu())

            metrics['test_correct'] += correct.item()
            metrics['test_loss_sum_over_batch'] += loss.item()
            metrics['test_total'] += target.size(0)

    # print(f"test_bs: {test_bs}")

    metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch'] / len(test_dataloader)
    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

    if verbose == 0:
        return metrics, None, None
    elif verbose == 1:
        return metrics, torch.cat(batch_predict_list), torch.cat(batch_label_list)


def test_given_dataloader_on_mix(model, test_dataloader, criterion, device=None, non_blocking=True, verbose=0):
    model.to(device, non_blocking=non_blocking)
    model.eval()

    metrics = {
        'test_correct': 0,
        'test_loss_sum_over_batch': 0,
        'test_total': 0,
    }

    criterion = criterion.to(device, non_blocking=non_blocking)

    if verbose == 1:
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

    with torch.no_grad():
        for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(test_dataloader):
            x = x.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            pred = model(x)
            loss = criterion(pred, labels.long())

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(labels).sum()

            if verbose == 1:
                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(labels.detach().clone().cpu())
                batch_original_index_list.append(original_index.detach().clone().cpu())
                batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                batch_original_targets_list.append(original_targets.detach().clone().cpu())

            metrics['test_correct'] += correct.item()
            metrics['test_loss_sum_over_batch'] += loss.item()
            metrics['test_total'] += labels.size(0)

    metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch'] / len(test_dataloader)
    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

    if verbose == 0:
        return metrics, \
            None, None, None, None, None
    elif verbose == 1:
        return metrics, \
            torch.cat(batch_predict_list), \
            torch.cat(batch_label_list), \
            torch.cat(batch_original_index_list), \
            torch.cat(batch_poison_indicator_list), \
            torch.cat(batch_original_targets_list)


def validate_list_for_plot(given_list, require_len=None):
    if (require_len is not None) and (len(given_list) == require_len):
        pass
    else:
        return False

    if None in given_list:
        return False

    return True


def general_plot_for_epoch(
        labelToListDict: dict,
        save_path: str,
        ylabel: str,
        xlabel: str = "epoch",
        y_min=None,
        y_max=None,
        title: str = "Results",
):
    # len of first list
    len_of_first_valueList = len(list(labelToListDict.values())[0])

    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = len(labelToListDict)
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6))  # 4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    # hese line of set linestyple is from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    linestyle_tuple = [
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),
        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    all_min = np.infty
    all_max = -np.infty
    for idx, (label, value_list) in enumerate(labelToListDict.items()):
        linestyle = linestyle_tuple[
            idx % len(linestyle_tuple)
            ][1]
        if validate_list_for_plot(value_list, len_of_first_valueList):
            plt.plot(range(len(value_list)), value_list, marker=idx % 11, linewidth=2, label=label, linestyle=linestyle)
        else:
            logging.warning(f"list:{label} contains None or len not match")
        once_min, once_max = min(value_list), max(value_list)
        all_min = once_min if once_min < all_min else all_min
        all_max = once_max if once_max > all_max else all_max

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.ylim(
        (all_min, all_max) if (y_min is None) or (y_max is None) else (float(y_min), float(y_max))
    )
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def plot_loss(
        train_loss_list: list,
        clean_test_loss_list: list,
        bd_test_loss_list: list,
        save_folder_path: str,
        save_file_name="loss_metric_plots",
):
    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = 3
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6))  # 4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    len_set = len(train_loss_list)
    x = range(len_set)
    if validate_list_for_plot(train_loss_list, len_set):
        plt.plot(x, train_loss_list, marker="o", linewidth=2, label="Train Loss", linestyle="--")
    else:
        logging.warning("train_loss_list contains None or len not match")
    if validate_list_for_plot(clean_test_loss_list, len_set):
        plt.plot(x, clean_test_loss_list, marker="v", linewidth=2, label="Test Clean loss", linestyle="-")
    else:
        logging.warning("clean_test_loss_list contains None or len not match")
    if validate_list_for_plot(bd_test_loss_list, len_set):
        plt.plot(x, bd_test_loss_list, marker="+", linewidth=2, label="Test Backdoor Loss", linestyle="-.")
    else:
        logging.warning("bd_test_loss_list contains None or len not match")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.ylim((0,
              max([value for value in  # filter None value
                   train_loss_list +
                   clean_test_loss_list +
                   bd_test_loss_list if value is not None])
              ))
    plt.legend()
    plt.title("Results")
    plt.grid()
    plt.savefig(f"{save_folder_path}/{save_file_name}.png")
    plt.close()


def plot_acc_like_metric_pure(
        train_acc_list: list,
        test_acc_list: list,
        test_asr_list: list,
        test_ra_list: list,
        save_folder_path: str,
        save_file_name="acc_like_metric_plots",
):
    len_set = len(test_asr_list)
    x = range(len(test_asr_list))

    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = 6
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6))  # 4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    if validate_list_for_plot(train_acc_list, len_set):
        plt.plot(x, train_acc_list, marker="o", linewidth=2, label="Train Acc", linestyle="--")
    else:
        logging.warning("train_acc_list contains None, or len not match")
    if validate_list_for_plot(test_acc_list, len_set):
        plt.plot(x, test_acc_list, marker="o", linewidth=2, label="Test C-Acc", linestyle="--")
    else:
        logging.warning("test_acc_list contains None, or len not match")
    if validate_list_for_plot(test_asr_list, len_set):
        plt.plot(x, test_asr_list, marker="v", linewidth=2, label="Test ASR", linestyle="-")
    else:
        logging.warning("test_asr_list contains None, or len not match")
    if validate_list_for_plot(test_ra_list, len_set):
        plt.plot(x, test_ra_list, marker="+", linewidth=2, label="Test RA", linestyle="-.")
    else:
        logging.warning("test_ra_list contains None, or len not match")

    plt.xlabel("Epochs")
    plt.ylabel("ACC")

    plt.ylim((0, 1))
    plt.legend()
    plt.title("Results")
    plt.grid()
    plt.savefig(f"{save_folder_path}/{save_file_name}.png")
    plt.close()


def plot_acc_like_metric(
        train_acc_list: list,
        train_asr_list: list,
        train_ra_list: list,
        test_acc_list: list,
        test_asr_list: list,
        test_ra_list: list,
        save_folder_path: str,
        save_file_name="acc_like_metric_plots",
):
    len_set = len(test_asr_list)
    x = range(len(test_asr_list))

    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = 6
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6))  # 4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    if validate_list_for_plot(train_acc_list, len_set):
        plt.plot(x, train_acc_list, marker="o", linewidth=2, label="Train Acc", linestyle="--")
    else:
        logging.warning("train_acc_list contains None, or len not match")
    if validate_list_for_plot(train_asr_list, len_set):
        plt.plot(x, train_asr_list, marker="v", linewidth=2, label="Train ASR", linestyle="-")
    else:
        logging.warning("train_asr_list contains None, or len not match")
    if validate_list_for_plot(train_ra_list, len_set):
        plt.plot(x, train_ra_list, marker="+", linewidth=2, label="Train RA", linestyle="-.")
    else:
        logging.warning("train_ra_list contains None, or len not match")
    if validate_list_for_plot(test_acc_list, len_set):
        plt.plot(x, test_acc_list, marker="o", linewidth=2, label="Test C-Acc", linestyle="--")
    else:
        logging.warning("test_acc_list contains None, or len not match")
    if validate_list_for_plot(test_asr_list, len_set):
        plt.plot(x, test_asr_list, marker="v", linewidth=2, label="Test ASR", linestyle="-")
    else:
        logging.warning("test_asr_list contains None, or len not match")
    if validate_list_for_plot(test_ra_list, len_set):
        plt.plot(x, test_ra_list, marker="+", linewidth=2, label="Test RA", linestyle="-.")
    else:
        logging.warning("test_ra_list contains None, or len not match")

    plt.xlabel("Epochs")
    plt.ylabel("ACC")

    plt.ylim((0, 1))
    plt.legend()
    plt.title("Results")
    plt.grid()
    plt.savefig(f"{save_folder_path}/{save_file_name}.png")
    plt.close()


class ModelTrainerCLS_v2():

    def __init__(self, model):
        self.model = model

    def set_with_dataloader(
            self,
            train_dataloader,
            test_dataloader_dict,

            criterion,
            optimizer,
            scheduler,
            device,
            amp,

            frequency_save,
            save_folder_path,
            save_prefix,

            prefetch=False,
            prefetch_transform_attr_name="transform",
            non_blocking=False,

            # continue_training_path: Optional[str] = None,
            # only_load_model: bool = False,
    ):

        logging.info(
            "Do NOT set the settings/parameters attr manually after you start training!" +
            "\nYou may break the relationship between them."
        )

        if non_blocking == False:
            logging.warning(
                "Make sure non_blocking=True if you use pin_memory or prefetch or other tricks depending on non_blocking."
            )

        self.train_dataloader = train_dataloader
        self.test_dataloader_dict = test_dataloader_dict

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.non_blocking = non_blocking

        self.frequency_save = frequency_save
        self.save_folder_path = save_folder_path
        self.save_prefix = save_prefix

        if prefetch:
            logging.debug("Converting dataloader to prefetch version.")

            train_dataset = self.train_dataloader.dataset
            train_prefetch_transform, train_mean, train_std = prefetch_transform(
                getattr(train_dataset, prefetch_transform_attr_name)
            )
            setattr(train_dataset, prefetch_transform_attr_name, train_prefetch_transform)
            self.train_dataloader = PrefetchLoader(
                self.train_dataloader, train_mean, train_std
            )
            for name, test_dataloader in self.test_dataloader_dict.items():
                val_dataset = test_dataloader.dataset
                val_prefetch_transform, val_mean, val_std = prefetch_transform(
                    getattr(val_dataset, prefetch_transform_attr_name)
                )
                setattr(val_dataset, prefetch_transform_attr_name, val_prefetch_transform)
                test_dataloader = PrefetchLoader(
                    test_dataloader, val_mean, val_std
                )
                self.test_dataloader_dict[name] = test_dataloader

        self.batch_num_per_epoch = len(self.train_dataloader)

        self.train_iter = iter(self.train_dataloader)

        # if continue_training_path is not None:
        #     logging.info(f"No batch info will be used. Cannot continue from specific batch!")
        #     self.epoch_now, self.batch_now = self.load_from_path(continue_training_path, device, only_load_model)
        #     assert self.batch_now < self.batch_num_per_epoch
        # else:
        self.epoch_now, self.batch_now = 0, 0

        logging.info(
            pformat(
                f"epoch_now:{self.epoch_now}, batch_now:{self.batch_now}" +
                f"self.amp:{self.amp}," +
                f"self.criterion:{self.criterion}," +
                f"self.optimizer:{self.optimizer}," +
                f"self.scheduler:{self.scheduler.state_dict() if self.scheduler is not None else None}," +
                f"self.scaler:{self.scaler.state_dict() if self.scaler is not None else None})"
            )
        )

        self.metric_aggregator = Metric_Aggregator()

        self.train_batch_loss_record = []

    def set_with_dataset(
            self,
            train_dataset,
            test_dataset_dict,

            batch_size,
            criterion,
            optimizer,
            scheduler,
            device,

            frequency_save,
            save_folder_path,
            save_prefix,

            amp=False,

            prefetch=True,
            prefetch_transform_attr_name="transform",
            non_blocking=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            num_workers=4,

            # continue_training_path: Optional[str] = None,
            # only_load_model: bool = False,
    ):

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
        )

        test_dataloader_dict = {
            name: DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
                num_workers=num_workers,
            )
            for name, test_dataset in test_dataset_dict.items()
        }

        self.set_with_dataloader(
            train_dataloader=train_dataloader,
            test_dataloader_dict=test_dataloader_dict,

            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp=amp,

            frequency_save=frequency_save,
            save_folder_path=save_folder_path,
            save_prefix=save_prefix,

            prefetch=prefetch,
            prefetch_transform_attr_name=prefetch_transform_attr_name,
            non_blocking=non_blocking,

            # continue_training_path = continue_training_path,
            # only_load_model = only_load_model,
        )

    def convert_to_batch_num(self, epochs=0, batchs=0):
        return int(epochs * self.batch_num_per_epoch + batchs)

    def get_one_batch(self):

        if self.batch_now == self.batch_num_per_epoch:

            self.epoch_now += 1
            self.batch_now = 0

            self.train_iter = iter(self.train_dataloader)

            if self.frequency_save != 0 and self.epoch_now % self.frequency_save == self.frequency_save - 1:
                logging.info(f'saved. epoch:{self.epoch_now}')
                self.save_all_state_to_path(
                    path=f"{self.save_folder_path}/{self.save_prefix}_epoch_{self.epoch_now}.pt")

            self.agg_save_dataframe()

        self.batch_now += 1

        return self.train_iter.__next__()

    def get_one_train_epoch_loss_avg_over_batch(self):
        if len(self.train_batch_loss_record) >= self.batch_num_per_epoch:
            return sum(
                self.train_batch_loss_record[-self.batch_num_per_epoch:]
            ) / self.batch_num_per_epoch
        else:
            logging.warning("No enough batch loss to get the one epoch loss")

    def one_forward_backward(self, x, labels, device, verbose=0):

        # t1 = time.time()

        self.model.train()
        self.model.to(device, non_blocking=self.non_blocking)

        x, labels = x.to(device, non_blocking=self.non_blocking), labels.to(device, non_blocking=self.non_blocking)
        bs = x.shape[0]
        # t2 = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp):
            image = x

            discrepancy = NuclearWassersteinDiscrepancy(self.model)
            # "Nuclear Wasserstein Discrepancy"值：通过计算两个矩阵分布的核范数之间的差异来衡量它们之间的相似度。
            # 核范数是一种矩阵的特征值的和，表示矩阵的所有奇异值之和。通过比较两个矩阵分布的核范数差异，可以得到它们在全局范围内的差异程度。

            if use_img_style_transfer:
                # style 1：图像域风格迁移
                tar0 = copy.deepcopy(image)
                if partial_transfer:
                    # for i in range(transfer_img_num):
                    #     tar0[i] = trans(image[i])  # 只对前 transfer_img_num 张图片进行图像迁移【串行】
                    x[:transfer_img_num] = trans(x[:transfer_img_num])
                else:
                    tar0 = trans(image)  # 对全部图片并行进行图像迁移
                x0 = torch.cat((image, tar0), dim=0)
                pred_clean0, mid_output0 = self.model.get_mid_forward(x0)
                pred_clean, predt0 = pred_clean0.chunk(2, dim=0)
                nwd0 = self.model.use_mid_nwdforward(mid_output0)
                # nwd0 = self.model.nwdforward(copy.deepcopy(x0))

                discrepancy_loss0 = -discrepancy(nwd0)
                distillation_loss0 = F.kl_div(
                    F.log_softmax(pred_clean / 1, dim=1),
                    F.log_softmax(predt0 / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / pred_clean0.numel()


            # style 2：噪声域风格迁移
            tar1 = copy.deepcopy(image)
            noise = torch.randn_like(tar1)
            tar1 = tar1 + noise_sigma * noise
            x1 = torch.cat((image, tar1), dim=0)

            # for preact_resnet
            # pred_clean1, mid_output1 = self.model.get_mid_forward(x1)   # get_mid_forward的目的是同时得到最终输出和中间层输出
            # pred_clean, predt1 = pred_clean1.chunk(2, dim=0)
            # nwd1 = self.model.use_mid_nwdforward(mid_output1)   # mid_output1: [bs,512,4,4]

            ### for vit model
            # vit model的中间层输出怎么获得？给该类增加一个方法
            pred_clean1, mid_output1 = self.model.vit_forword(self.model[1], x1)    # mid_output1: [bs*2, 768]
            pred_clean, predt1 = pred_clean1.chunk(2, dim=0)
            # nwd1 = self.model.use_mid_nwdforward(self.model[1], mid_output1)
            # nwd1 = mid_output1.reshape(bs*2, 3, 16, 16)
            nwd1 = mid_output1.reshape(bs*2, 3, -1, 32)
            nwd1 = nwd1.repeat(1,1,4,1)
            # 后面计算discrepancy_loss时，会调用StyTR.py中的encode_with_intermediate函数，其中分别使用5层vgg网络进行卷积，
            # 然而当最后两个维度中的任意一维变成1时，pad(1,1)会报错，因此后两个维度(H和W)需要足够大
            # Q：在resnet架构中，nwd的shape是多少？

            # mid_output1_l, mid_output1_r = mid_output1.chunk(2, dim=0)
            # print("+" * 10)
            # print(f"pred_clean: {pred_clean}")
            # print(f"predt1: {predt1}")    # predt1 和 pred_clean 是一样的
            # print(f"mid_output1_l: {mid_output1_l}")
            # print(f"mid_output1_r: {mid_output1_r}")
            # print(f"equal: {torch.equal(mid_output1_l, mid_output1_r)}")

            # nwd1 = self.model.forward(mid_output1)
            # nwd1 = self.model.nwdforward(copy.deepcopy(x1))
            # nwd1_2 = self.model.nwdforward(copy.deepcopy(x1))
            # print(f"equal: {torch.equal(nwd1, nwd1_2)}")
            # print(f"nwd1: {nwd1}")
            # print(f"nwd1_2: {nwd1_2}")

            # 风格迁移loss: 让风格的特征层的分布和clean靠拢
            assert len(nwd1.shape) == 4 # 必须是四维的
            discrepancy_loss1 = -discrepancy(nwd1)
            # 知识蒸馏loss: 让风格迁移的pred学会clean的知识
            # We provide the teacher's targets in log probability because we use log_target=True
            # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
            # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
            distillation_loss1 = F.kl_div(
                F.log_softmax(pred_clean / 1, dim=1),
                F.log_softmax(predt1 / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / pred_clean1.numel()

            ### Loss ###
            # 原样本的分类loss
            cls_loss = self.criterion(pred_clean, labels)

            # 迁移loss 和 蒸馏loss
            transfer_lamda = 1e-7 # 风格超参
            if use_img_style_transfer:
                transfer_loss = 0.5 * (discrepancy_loss0 + discrepancy_loss1) * transfer_lamda
                distillation_loss = 0.5 * (distillation_loss0 + distillation_loss1)
            else:
                transfer_loss = 1 * (discrepancy_loss1) * transfer_lamda
                distillation_loss = 1 * (distillation_loss1)

            loss = cls_loss * (1 - alpha) + distillation_loss * alpha + transfer_loss   # alpha: 蒸馏超参
            # loss = cls_loss   # alpha: 蒸馏超参

            # t8 = time.time()

            # if verbose:
            #     log_probs = self.model(tar0)
            # loss = self.criterion(log_probs, labels.long())

        # print("=" * 10)
        # print(f"in one_forward_backward loss:{loss}")
        # print(f"in one_forward_backward cls_loss:{cls_loss}")
        # print(f"in one_forward_backward distillation_loss:{distillation_loss}")
        # print(f"in one_forward_backward transfer_loss:{transfer_loss} \n\n") # transfer_loss有可能是-inf

        self.scaler.scale(loss).backward()

        # t9 = time.time()

        self.scaler.step(self.optimizer)

        # t10 = time.time()


        self.scaler.update()
        self.optimizer.zero_grad()
        batch_loss = loss.item()

        # t11 = time.time()

        # if show_progress:
        #     timelist = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11]
        #     logging.info("=" * 10)
        #     logging.info(f"\n Time consuming:")
        #     for i in range(len(timelist) - 1):
        #         logging.info(f'{i+1}~{i+2} cost: {timelist[i + 1] - timelist[i]:.5f}')
        #     logging.info("=" * 10)
        #     logging.info(f" === batch time: {timelist[-1] - timelist[0]}")

        # if verbose == 1:
        #     batch_predict = torch.max(log_probs, -1)[1].detach().clone().cpu()
        #     return batch_loss, batch_predict

        if use_img_style_transfer:
            batch_predict = torch.max(predt0, -1)[1].detach().clone().cpu()
        else:
            batch_predict = torch.max(predt1, -1)[1].detach().clone().cpu()
        return batch_loss, batch_predict

    def train(self, epochs=0, batchs=0):

        train_batch_num = self.convert_to_batch_num(epochs, batchs)

        for idx in range(train_batch_num):

            x, labels, *additional_info = self.get_one_batch()
            batch_loss, _ = self.one_forward_backward(x, labels, self.device)

            self.train_batch_loss_record.append(batch_loss)

            if self.batch_now == 0 and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # here since ReduceLROnPlateau need the train loss to decide next step setting.
                    self.scheduler.step(self.get_one_train_epoch_loss_avg_over_batch())
                else:
                    self.scheduler.step()

    def test_given_dataloader(self, test_dataloader, device=None, verbose=0):

        if device is None:
            device = self.device

        model = self.model
        non_blocking = self.non_blocking

        return given_dataloader_test(
            model,
            test_dataloader,
            self.criterion,
            non_blocking,
            device,
            verbose,
        )

    def test_all_inner_dataloader(self):
        metrics_dict = {}
        for name, test_dataloader in self.test_dataloader_dict.items():
            metrics_dict[name], *other_returns = self.test_given_dataloader(
                test_dataloader,
                verbose=1,
            )
        return metrics_dict

    def agg(self, info_dict):
        info = {
            "epoch": self.epoch_now,
            "batch": self.batch_now,
        }
        info.update(info_dict)
        self.metric_aggregator(
            info
        )

    def train_one_epoch(self, verbose=0):

        startTime = time.time()

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []

        for batch_idx in range(self.batch_num_per_epoch):
            x, labels, *additional_info = self.get_one_batch()
            one_batch_loss, batch_predict = self.one_forward_backward(x, labels, self.device, verbose)
            batch_loss_list.append(one_batch_loss)

            batch_predict_list.append(batch_predict.detach().clone().cpu())
            batch_label_list.append(labels.detach().clone().cpu())

        train_one_epoch_loss_batch_avg = sum(batch_loss_list) / len(batch_loss_list)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_one_epoch_loss_batch_avg)
            else:
                self.scheduler.step()

        endTime = time.time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        return train_one_epoch_loss_batch_avg, torch.cat(batch_predict_list), torch.cat(batch_label_list)

    def train_with_test_each_epoch(self,
                                   train_dataloader,
                                   test_dataloader_dict,
                                   total_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   amp,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   prefetch,
                                   prefetch_transform_attr_name,
                                   non_blocking,
                                   ):

        self.set_with_dataloader(
            train_dataloader,
            test_dataloader_dict,

            criterion,
            optimizer,
            scheduler,
            device,
            amp,

            frequency_save,
            save_folder_path,
            save_prefix,

            prefetch,
            prefetch_transform_attr_name,
            non_blocking,

            # continue_training_path,
            # only_load_model,
        )

        for epoch in range(total_epoch_num):

            logging.info(f" train epoch {epoch}/{total_epoch_num}")

            train_one_epoch_loss_batch_avg, train_epoch_predict_list, train_epoch_label_list = self.train_one_epoch(
                verbose=1)

            info_dict_for_one_epoch = {}
            info_dict_for_one_epoch.update(
                {
                    "train_epoch_loss_avg_over_batch": train_one_epoch_loss_batch_avg,
                    "train_acc": all_acc(train_epoch_predict_list, train_epoch_label_list),
                }
            )

            for dataloader_name, test_dataloader in test_dataloader_dict.items():
                metrics, *other_returns = self.test_given_dataloader(test_dataloader)
                info_dict_for_one_epoch.update(
                    {
                        f"{dataloader_name}_{k}": v for k, v in metrics.items()
                    }
                )

            self.agg(info_dict_for_one_epoch)

        self.agg_save_summary()

    def agg_save_dataframe(self):
        self.metric_aggregator.to_dataframe().to_csv(f"{self.save_folder_path}/{self.save_prefix}_df.csv")

    def agg_save_summary(self):
        self.metric_aggregator.summary().to_csv(f"{self.save_folder_path}/{self.save_prefix}_df_summary.csv")

    def get_model_params(self):
        return self.model.cpu().state_dict()

    # def set_model_params(self, model_parameters):
    #     self.model.load_state_dict(model_parameters)

    def save_all_state_to_path(self,
                               path: str,
                               only_model_state_dict: bool = False) -> None:
        '''
        save all information needed to continue training, include 3 random state in random, numpy and torch
        :param path: where to save
        :param epoch: which epoch when save
        :param batch: which batch index when save
        :param only_model_state_dict: only save the model, drop all other information
        '''

        epoch, batch = self.epoch_now, self.batch_now

        save_dict = {
            'epoch_num_when_save': epoch,
            'batch_num_when_save': batch,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'model_state_dict': self.get_model_params(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
            "scaler": self.scaler.state_dict(),
        } \
            if only_model_state_dict == False else self.get_model_params()

        torch.save(
            save_dict,
            path,
        )

    # def load_from_path(self,
    #                    path: str,
    #                    device,
    #                    only_load_model: bool = False
    #                    ) -> [Optional[int], Optional[int]]:
    #     '''
    #
    #     :param path:
    #     :param device: map model to which device
    #     :param only_load_model: only_load_model or not?
    #     '''
    #
    #     self.model = self.model.to(device, non_blocking=self.non_blocking)
    #
    #     load_dict = torch.load(
    #         path, map_location=device
    #     )
    #
    #     logging.info(f"loading... keys:{load_dict.keys()}, only_load_model:{only_load_model}")
    #
    #     attr_list = [
    #         'epoch_num_when_save',
    #         'batch_num_when_save',
    #         'random_state',
    #         'np_random_state',
    #         'torch_random_state',
    #         'model_state_dict',
    #         'optimizer_state_dict',
    #         'scheduler_state_dict',
    #         'criterion_state_dict',
    #     ]
    #
    #     if all([key_name in load_dict for key_name in attr_list]) :
    #         # all required key can find in load dict
    #         # AND only_load_model == False
    #         if only_load_model == False:
    #             random.setstate(load_dict['random_state'])
    #             np.random.set_state(load_dict['np_random_state'])
    #             torch.random.set_rng_state(load_dict['torch_random_state'].cpu()) # since may map to cuda
    #
    #             self.model.load_state_dict(
    #                 load_dict['model_state_dict']
    #             )
    #             self.optimizer.load_state_dict(
    #                 load_dict['optimizer_state_dict']
    #             )
    #             if self.scheduler is not None:
    #                 self.scheduler.load_state_dict(
    #                     load_dict['scheduler_state_dict']
    #                 )
    #             self.criterion.load_state_dict(
    #                 load_dict['criterion_state_dict']
    #             )
    #             if 'scaler' in load_dict:
    #                 self.scaler.load_state_dict(
    #                     load_dict["scaler"]
    #                 )
    #                 logging.info(f'load scaler done. scaler={load_dict["scaler"]}')
    #             logging.info('all state load successful')
    #             return load_dict['epoch_num_when_save'], load_dict['batch_num_when_save']
    #         else:
    #             self.model.load_state_dict(
    #                 load_dict['model_state_dict'],
    #             )
    #             logging.info('only model state_dict load')
    #             return None, None
    #
    #     else:  # only state_dict
    #
    #         if 'model_state_dict' in load_dict:
    #             self.model.load_state_dict(
    #                 load_dict['model_state_dict'],
    #             )
    #             logging.info('only model state_dict load')
    #             return None, None
    #         else:
    #             self.model.load_state_dict(
    #                 load_dict,
    #             )
    #             logging.info('only model state_dict load')
    #             return None, None
    #


class PureCleanModelTrainer(ModelTrainerCLS_v2):

    def __init__(self, model):
        super().__init__(model)
        logging.debug(
            "This class REQUIRE bd dataset to implement overwrite methods. This is NOT a general class for all cls task.")
        self.train_time = 0

    def train_one_epoch_on_mix(self, verbose=0):

        startTime = time.time()

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        # if verbose == 1:
        # batch_original_index_list = []
        # batch_poison_indicator_list = []
        # batch_original_targets_list = []

        for batch_idx in range(self.batch_num_per_epoch):
            if show_progress:
                logging.info(f" train batch {batch_idx}/{self.batch_num_per_epoch}")
            # if batch_idx <= self.batch_num_per_epoch - 2:     # for debug
            #     continue

            # x, labels, original_index, poison_indicator, original_targets  = self.get_one_batch()
            x, labels = self.get_one_batch()

            s = time.time()
            one_batch_loss, batch_predict = self.one_forward_backward(x, labels, self.device, verbose)
            # print(f"one_batch_loss: {one_batch_loss}")
            e = time.time()
            self.train_time += e-s

            batch_loss_list.append(one_batch_loss)

            batch_predict_list.append(batch_predict.detach().clone().cpu())
            batch_label_list.append(labels.detach().clone().cpu())
            # if verbose == 1:
            # batch_original_index_list.append(original_index.detach().clone().cpu())
            # batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            # batch_original_targets_list.append(original_targets.detach().clone().cpu())

        one_epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        # print(f"sum(batch_loss_list): {sum(batch_loss_list)}, len(batch_loss_list: {len(batch_loss_list)}, one_epoch_loss: {one_epoch_loss}")
        # 为什么结果会有-inf？？？

        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time.time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        if verbose == 0:
            return one_epoch_loss, \
                torch.cat(batch_predict_list), \
                torch.cat(batch_label_list), \
                None, None, None
        elif verbose == 1:
            return one_epoch_loss, \
                torch.cat(batch_predict_list), \
                torch.cat(batch_label_list)
            # torch.cat(batch_original_index_list), \
            # torch.cat(batch_poison_indicator_list), \
            # torch.cat(batch_original_targets_list)

    def test_given_dataloader_on_mix(self, test_dataloader, device=None, verbose=0):

        if device is None:
            device = self.device

        model = self.model
        model.to(device, non_blocking=self.non_blocking)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss_sum_over_batch': 0,
            'test_total': 0,
        }

        criterion = self.criterion.to(device, non_blocking=self.non_blocking)
        batch_predict_list = []
        batch_label_list = []
        batch_original_targets_list = []
        if verbose == 1:
            batch_original_index_list = []
            batch_poison_indicator_list = []

        with torch.no_grad():
            for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(
                    test_dataloader):
                x = x.to(device, non_blocking=self.non_blocking)
                labels = labels.to(device, non_blocking=self.non_blocking)

                if use_img_style_transfer:
                    # 图像迁移
                    if partial_transfer:
                        # for i in range(transfer_img_num):
                        #     tar0[i] = trans(x[i])  # 只对前 transfer_img_num 张图片进行图像迁移
                        x[:transfer_img_num] = trans(x[:transfer_img_num])  # 并行
                    else:
                        x = trans(x)  # 对全部图片并行进行图像迁移

                pred = model(x)
                loss = criterion(pred, labels.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(labels).sum()

                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(labels.detach().clone().cpu())
                batch_original_targets_list.append(original_targets.detach().clone().cpu())
                if verbose == 1:
                    batch_original_index_list.append(original_index.detach().clone().cpu())
                    batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())

                metrics['test_correct'] += correct.item()
                metrics['test_loss_sum_over_batch'] += loss.item()
                metrics['test_total'] += labels.size(0)

        metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch'] / len(test_dataloader)
        metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

        if verbose == 0:
            return metrics, \
                torch.cat(batch_predict_list), \
                torch.cat(batch_label_list), \
                None, None, \
                torch.cat(batch_original_targets_list)
        elif verbose == 1:
            return metrics, \
                torch.cat(batch_predict_list), \
                torch.cat(batch_label_list), \
                torch.cat(batch_original_index_list), \
                torch.cat(batch_poison_indicator_list), \
                torch.cat(batch_original_targets_list)

    # ft train
    def train_with_test_each_epoch_on_mix(self,
                                          train_dataloader,
                                          clean_test_dataloader,
                                          bd_test_dataloader,
                                          total_epoch_num,
                                          criterion,
                                          optimizer,
                                          scheduler,
                                          amp,
                                          device,
                                          frequency_save,
                                          save_folder_path,
                                          save_prefix,
                                          prefetch,
                                          prefetch_transform_attr_name,
                                          non_blocking,
                                          ):

        global transfer_img_num
        global partial_transfer
        global use_img_style_transfer
        test_dataloader_dict = {
            "clean_test_dataloader": clean_test_dataloader,
            "bd_test_dataloader": bd_test_dataloader,
        }

        self.set_with_dataloader(
            train_dataloader,
            test_dataloader_dict,
            criterion,
            optimizer,
            scheduler,
            device,
            amp,

            frequency_save,
            save_folder_path,
            save_prefix,

            prefetch,
            prefetch_transform_attr_name,
            non_blocking,
        )

        train_loss_list = []
        train_mix_acc_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        for epoch in range(total_epoch_num):
            logging.info(f" === train epoch {epoch}/{total_epoch_num}, transfer_img_num: {transfer_img_num} === ")
            # train_epoch_loss_avg_over_batch, \
            # train_epoch_predict_list, \
            # train_epoch_label_list, \
            # train_epoch_original_index_list, \
            # train_epoch_poison_indicator_list, \
            # train_epoch_original_targets_list = self.train_one_epoch_on_mix(verbose=1)
            train_epoch_loss_avg_over_batch, \
                train_epoch_predict_list, \
                train_epoch_label_list = self.train_one_epoch_on_mix(verbose=1)
            logging.info(f" >>>> train_time: {self.train_time}, lr: {self.optimizer.param_groups[0]['lr']}")

            train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

            # train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
            # train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]

            test_start = time.time()
            logging.info("testing......")
            clean_metrics, \
                clean_test_epoch_predict_list, \
                clean_test_epoch_label_list, \
                = self.test_given_dataloader(test_dataloader_dict["clean_test_dataloader"], verbose=1)
            test_end = time.time()
            logging.info(f"test cost: {test_end-test_start}")

            clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
            test_acc = clean_metrics["test_acc"]

            test_start = time.time()
            bd_metrics, \
                bd_test_epoch_predict_list, \
                bd_test_epoch_label_list, \
                bd_test_epoch_original_index_list, \
                bd_test_epoch_poison_indicator_list, \
                bd_test_epoch_original_targets_list = self.test_given_dataloader_on_mix(
                test_dataloader_dict["bd_test_dataloader"], verbose=1)
            test_end = time.time()
            logging.info(f"test mix cost: {test_end - test_start}")

            bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
            test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

            # adaptive adjust transfer_img_num
            # if bd_test_loss_avg_over_batch < 1:
            #     transfer_img_num = max(transfer_img_num, 1)
            # elif bd_test_loss_avg_over_batch < 3:
            #     transfer_img_num = max(transfer_img_num, 3)
            # elif bd_test_loss_avg_over_batch < 5:
            #     transfer_img_num = max(transfer_img_num, 5)
            # elif bd_test_loss_avg_over_batch < 7:
            #     transfer_img_num = max(transfer_img_num, 7)
            # elif bd_test_loss_avg_over_batch < 9:
            #     transfer_img_num = max(transfer_img_num, 9)
            # elif bd_test_loss_avg_over_batch < 11:
            #     transfer_img_num = max(transfer_img_num, 15)
            # elif bd_test_loss_avg_over_batch < 13:
            #     use_img_style_transfer = True
            #     partial_transfer = False
            transfer_img_num = int(np.exp(bd_test_loss_avg_over_batch))


            # 这里计算出来的acc，asr，ra就是打印出来的结果。
            # print(f" === test_acc: {test_acc} ===")
            # print(f" === test_asr: {test_asr} ===")
            # print(f" === test_ra: {test_ra} ===")
            # 为什么acc，ra会这么低？？

            if test_asr <= 0.2:
                print(f"defense success!!! train time: {self.train_time}")

            self.agg(
                {
                    "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                    "train_acc": train_mix_acc,

                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                }
            )

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            self.plot_loss(
                train_loss_list,
                clean_test_loss_list,
                bd_test_loss_list,
            )

            self.plot_acc_like_metric(
                train_mix_acc_list,
                test_acc_list,
                test_asr_list,
                test_ra_list,
            )

            self.agg_save_dataframe()

        self.agg_save_summary()

        return train_loss_list, \
            train_mix_acc_list, \
            clean_test_loss_list, \
            bd_test_loss_list, \
            test_acc_list, \
            test_asr_list, \
            test_ra_list

    def plot_loss(
            self,
            train_loss_list: list,
            clean_test_loss_list: list,
            bd_test_loss_list: list,
            save_file_name="loss_metric_plots",
    ):

        plot_loss(
            train_loss_list,
            clean_test_loss_list,
            bd_test_loss_list,
            self.save_folder_path,
            save_file_name,
        )

    def plot_acc_like_metric(self,
                             train_acc_list: list,
                             test_acc_list: list,
                             test_asr_list: list,
                             test_ra_list: list,
                             save_file_name="acc_like_metric_plots",
                             ):

        plot_acc_like_metric_pure(
            train_acc_list,
            test_acc_list,
            test_asr_list,
            test_ra_list,
            self.save_folder_path,
            save_file_name,
        )

    def test_current_model(self, test_dataloader_dict, device=None, ):

        if device is None:
            device = self.device

        model = self.model
        model.to(device, non_blocking=self.non_blocking)
        model.eval()

        clean_metrics, \
            clean_test_epoch_predict_list, \
            clean_test_epoch_label_list, \
            = self.test_given_dataloader(test_dataloader_dict["clean_test_dataloader"], verbose=1)

        clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
        test_acc = clean_metrics["test_acc"]

        bd_metrics, \
            bd_test_epoch_predict_list, \
            bd_test_epoch_label_list, \
            bd_test_epoch_original_index_list, \
            bd_test_epoch_poison_indicator_list, \
            bd_test_epoch_original_targets_list = self.test_given_dataloader_on_mix(
            test_dataloader_dict["bd_test_dataloader"], verbose=1)

        bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
        test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
        test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

        return clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra


class BackdoorModelTrainer(ModelTrainerCLS_v2):

    def __init__(self, model):
        super().__init__(model)
        logging.debug(
            "This class REQUIRE bd dataset to implement overwrite methods. This is NOT a general class for all cls task.")

    def train_one_epoch_on_mix(self, verbose=0):

        startTime = time.time()

        batch_loss_list = []
        if verbose == 1:
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []

        for batch_idx in range(self.batch_num_per_epoch):
            x, labels, original_index, poison_indicator, original_targets = self.get_one_batch()
            one_batch_loss, batch_predict = self.one_forward_backward(x, labels, self.device, verbose)
            batch_loss_list.append(one_batch_loss)

            if verbose == 1:
                batch_predict_list.append(batch_predict.detach().clone().cpu())
                batch_label_list.append(labels.detach().clone().cpu())
                batch_original_index_list.append(original_index.detach().clone().cpu())
                batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                batch_original_targets_list.append(original_targets.detach().clone().cpu())

        one_epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time.time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        if verbose == 0:
            return one_epoch_loss, \
                None, None, None, None, None
        elif verbose == 1:
            return one_epoch_loss, \
                torch.cat(batch_predict_list), \
                torch.cat(batch_label_list), \
                torch.cat(batch_original_index_list), \
                torch.cat(batch_poison_indicator_list), \
                torch.cat(batch_original_targets_list)

    def test_given_dataloader_on_mix(self, test_dataloader, device=None, verbose=0):

        if device is None:
            device = self.device

        model = self.model
        model.to(device, non_blocking=self.non_blocking)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss_sum_over_batch': 0,
            'test_total': 0,
        }

        criterion = self.criterion.to(device, non_blocking=self.non_blocking)

        if verbose == 1:
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []

        with torch.no_grad():
            for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(
                    test_dataloader):
                x = x.to(device, non_blocking=self.non_blocking)
                labels = labels.to(device, non_blocking=self.non_blocking)
                pred = model(x)
                loss = criterion(pred, labels.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(labels).sum()

                if verbose == 1:
                    batch_predict_list.append(predicted.detach().clone().cpu())
                    batch_label_list.append(labels.detach().clone().cpu())
                    batch_original_index_list.append(original_index.detach().clone().cpu())
                    batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                    batch_original_targets_list.append(original_targets.detach().clone().cpu())

                metrics['test_correct'] += correct.item()
                metrics['test_loss_sum_over_batch'] += loss.item()
                metrics['test_total'] += labels.size(0)

        metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch'] / len(test_dataloader)
        metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

        if verbose == 0:
            return metrics, \
                None, None, None, None, None
        elif verbose == 1:
            return metrics, \
                torch.cat(batch_predict_list), \
                torch.cat(batch_label_list), \
                torch.cat(batch_original_index_list), \
                torch.cat(batch_poison_indicator_list), \
                torch.cat(batch_original_targets_list)

    def train_with_test_each_epoch_on_mix(self,
                                          train_dataloader,
                                          clean_test_dataloader,
                                          bd_test_dataloader,
                                          total_epoch_num,
                                          criterion,
                                          optimizer,
                                          scheduler,
                                          amp,
                                          device,
                                          frequency_save,
                                          save_folder_path,
                                          save_prefix,
                                          prefetch,
                                          prefetch_transform_attr_name,
                                          non_blocking,
                                          ):

        test_dataloader_dict = {
            "clean_test_dataloader": clean_test_dataloader,
            "bd_test_dataloader": bd_test_dataloader,
        }

        self.set_with_dataloader(
            train_dataloader,
            test_dataloader_dict,
            criterion,
            optimizer,
            scheduler,
            device,
            amp,

            frequency_save,
            save_folder_path,
            save_prefix,

            prefetch,
            prefetch_transform_attr_name,
            non_blocking,
        )

        train_loss_list = []
        train_mix_acc_list = []
        train_asr_list = []
        train_ra_list = []
        clean_test_loss_list = []
        bd_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        for epoch in range(total_epoch_num):
            train_epoch_loss_avg_over_batch, \
                train_epoch_predict_list, \
                train_epoch_label_list, \
                train_epoch_original_index_list, \
                train_epoch_poison_indicator_list, \
                train_epoch_original_targets_list = self.train_one_epoch_on_mix(verbose=1)

            train_mix_acc = all_acc(train_epoch_predict_list, train_epoch_label_list)

            train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
            train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
            train_clean_acc = all_acc(
                train_epoch_predict_list[train_clean_idx],
                train_epoch_label_list[train_clean_idx],
            )
            train_asr = all_acc(
                train_epoch_predict_list[train_bd_idx],
                train_epoch_label_list[train_bd_idx],
            )
            train_ra = all_acc(
                train_epoch_predict_list[train_bd_idx],
                train_epoch_original_targets_list[train_bd_idx],
            )

            clean_metrics, \
                clean_test_epoch_predict_list, \
                clean_test_epoch_label_list, \
                = self.test_given_dataloader(self.test_dataloader_dict["clean_test_dataloader"], verbose=1)

            clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
            test_acc = clean_metrics["test_acc"]

            bd_metrics, \
                bd_test_epoch_predict_list, \
                bd_test_epoch_label_list, \
                bd_test_epoch_original_index_list, \
                bd_test_epoch_poison_indicator_list, \
                bd_test_epoch_original_targets_list = self.test_given_dataloader_on_mix(
                self.test_dataloader_dict["bd_test_dataloader"], verbose=1)

            bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
            test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
            test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

            self.agg(
                {
                    "train_epoch_loss_avg_over_batch": train_epoch_loss_avg_over_batch,
                    "train_acc": train_mix_acc,
                    "train_acc_clean_only": train_clean_acc,
                    "train_asr_bd_only": train_asr,
                    "train_ra_bd_only": train_ra,

                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                }
            )

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_mix_acc_list.append(train_mix_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            self.plot_loss(
                train_loss_list,
                clean_test_loss_list,
                bd_test_loss_list,
            )

            self.plot_acc_like_metric(
                train_mix_acc_list,
                train_asr_list,
                train_ra_list,
                test_acc_list,
                test_asr_list,
                test_ra_list,
            )

            self.agg_save_dataframe()

        self.agg_save_summary()

        return train_loss_list, \
            train_mix_acc_list, \
            train_asr_list, \
            train_ra_list, \
            clean_test_loss_list, \
            bd_test_loss_list, \
            test_acc_list, \
            test_asr_list, \
            test_ra_list

    def plot_loss(
            self,
            train_loss_list: list,
            clean_test_loss_list: list,
            bd_test_loss_list: list,
            save_file_name="loss_metric_plots",
    ):

        plot_loss(
            train_loss_list,
            clean_test_loss_list,
            bd_test_loss_list,
            self.save_folder_path,
            save_file_name,
        )

    def plot_acc_like_metric(self,
                             train_acc_list: list,
                             train_asr_list: list,
                             train_ra_list: list,
                             test_acc_list: list,
                             test_asr_list: list,
                             test_ra_list: list,
                             save_file_name="acc_like_metric_plots",
                             ):

        plot_acc_like_metric(
            train_acc_list,
            train_asr_list,
            train_ra_list,
            test_acc_list,
            test_asr_list,
            test_ra_list,
            self.save_folder_path,
            save_file_name,
        )
