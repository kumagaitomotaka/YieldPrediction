import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test_for_ORD import MolTestDatasetWrapper
from sklearn.preprocessing import label_binarize

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune_for_ORD.yaml', os.path.join(model_checkpoints_folder, 'config_finetune_for_ORD.yaml'))

class Make_h(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('finetune_for_ORD', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        h, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return h, loss

    def pick_h(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        
        #モデルの読み込み
        if self.config['model_type'] == 'gin':
            from models.ginet_finetune_for_ORD import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"],ORD=True).to(self.device)
            if self.config['fine_tune_from'] != 'untrained_gin':
                model = self._load_pre_trained_weights(model) #事前学習データ
              
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            if self.config['fine_tune_from'] != 'untrained_gcn':
                model = self._load_pre_trained_weights(model) #事前学習データ

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0 
        
        train_h = []
        train_labels = []
        
        #only 1 cycle
        train_loss = 0
        num_train_data = 0
        for bn, data in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(self.device)
            h, loss = self._step(model, data, n_iter)
            
            if self.device == 'cpu':
                train_h.extend(h.detach().numpy())
                train_labels.extend(data.y.flatten().numpy())
            else:
                train_h.extend(h.cpu().detach().numpy())
                train_labels.extend(data.y.cpu().flatten().numpy())
                
            train_loss += loss.item() * data.y.size(0)
            num_train_data += data.y.size(0)

            if n_iter % self.config['log_every_n_steps'] == 0:
                    #self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(bn, loss.item())

            if apex_support and self.config['fp16_precision']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            n_iter += 1

        # validation data
        train_loss /= num_train_data
        print('Train loss:', train_loss)
        if self.config['dataset']['task'] == 'classification': 
            valid_h, valid_labels = self._validate(model, valid_loader)
        elif self.config['dataset']['task'] == 'regression': 
            valid_h, valid_labels = self._validate(model, valid_loader)

        #self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
        valid_n_iter += 1
        
        if self.config['CV'] or self.config['ORD_train']:
            return train_h, train_labels, valid_h, valid_labels, model
        else:
            #test data
            test_h, test_labels = self._test(model, test_loader)

            h_set = {'train': train_h, 'valid': valid_h, 'test': test_h}
            label_set = {'train': train_labels, 'valid': valid_labels, 'test': test_labels}

            return h_set, label_set

    def _load_pre_trained_weights(self, model):
        try:
            if self.config['fine_tune_from'] == 'pretrained_gin' or self.config['fine_tune_from'] == 'pretrained_gcn':
                checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'])
            elif self.config['fine_tune_from'] == 'ORD_rgr_trained_gin':
                checkpoints_folder = os.path.join('./log', 'untrained_gin_finetune_for_ORD_rgr', 'Mar02_15-13-23_ORD_rgr_yield','checkpoints')
            elif self.config['fine_tune_from'] == 'ORD_cl_trained_gin':
                checkpoints_folder = os.path.join('./log', 'untrained_gin_finetune_for_ORD', 'Mar06_11-44-18_ORD_yield_round','checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        valid_h = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                h, pred = model(data)
                _, loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    valid_h.extend(h.detach().numpy())
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    valid_h.extend(h.cpu().detach().numpy())
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_h, labels
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_h, labels

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            label_class = np.array(range(11))
            one_hot_labels = label_binarize(labels, classes=label_class)
            try:
                roc_auc = roc_auc_score(one_hot_labels, predictions, multi_class='ovr')
            except ValueError:
                roc_auc = 'could not calculate'
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_h,labels

    def _test(self, model, test_loader):
        #model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        #state_dict = torch.load(model_path, map_location=self.device)
        #model.load_state_dict(state_dict)
        #print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        test_h = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                h, pred = model(data)
                _, loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    test_h.extend(h.detach().numpy())
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    test_h.extend(h.cpu().detach().numpy())
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            label_class = np.array(range(11))
            one_hot_labels = label_binarize(labels, classes=label_class)
            try:
                self.roc_auc = roc_auc_score(one_hot_labels, predictions, multi_class='ovr')
            except ValueError:
                self.roc_auc = 'could not calculate'
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
        return test_h, labels