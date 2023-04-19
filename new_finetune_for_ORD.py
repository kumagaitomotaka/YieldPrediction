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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test_for_ORD import MolTestDatasetWrapper,read_smiles_for_ORD
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
import copy


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


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_file_dir = os.path.join('log','{}_finetune_for_{}'.format(config['fine_tune_from'], config['task_name']))
        os.makedirs(log_file_dir, exist_ok=True)
        log_dir = os.path.join(log_file_dir, dir_name)
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
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

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
        
        train_loss_list = []
        valid_loss_list = []
        epoch_list = []

        for epoch_counter in range(self.config['epochs']):
            train_loss = 0
            num_train_data = 0
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)
                
                train_loss += loss.item() * data.y.size(0)
                num_train_data += data.y.size(0)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                train_loss /= num_train_data
                print('Train loss:', train_loss)
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                epoch_list.append(epoch_counter)
        
        if self.config['CV'] or self.config['ORD_train']:
            return epoch_list,train_loss_list,valid_loss_list,model
        else:
            predictions,labels = self._test(model, test_loader)

            return epoch_list,train_loss_list,valid_loss_list,predictions,labels

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
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
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
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            if config['task_name'] == 'PC':
                max_pred = np.argmax(predictions, axis=1)
                accuracy = accuracy_score(labels, max_pred)
                return valid_loss, accuracy
            else:
                label_class = np.array(range(11))
                one_hot_labels = label_binarize(labels, classes=label_class)
                roc_auc = roc_auc_score(one_hot_labels, predictions, multi_class='ovr')
                print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
                return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
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
            if config['task_name'] == 'PC':
                max_pred = np.argmax(predictions, axis=1)
                self.accuracy = accuracy_score(labels, max_pred)
                print('Test loss:', test_loss, 'Test accuracy:', self.accuracy)
            else:
                label_class = np.array(range(11))
                one_hot_labels = label_binarize(labels, classes=label_class)
                self.roc_auc = roc_auc_score(one_hot_labels, predictions, multi_class='ovr')
                print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
        return predictions,labels
    
def save_calc_results(epoch_list,train_loss_list,valid_loss_list,predictions,labels,n=None):
    current_time = datetime.now().strftime('%y%m%d')
    if config['model_type'] == 'gin':
        os.makedirs('GIN_data', exist_ok=True)
        model_data_dir = 'GIN_data'
    elif config['model_type'] == 'gcn':
        os.makedirs('GCN_data', exist_ok=True)
        model_data_dir = 'GCN_data'
    else:
        raise ValueError('Undefined model type!')
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        rxn_name = ''
        for i,rtype in enumerate(config['rxn_type']):
            if i > 0:
                rxn_name += '-'
            rxn_name += rtype
        if config['ORD_train']:
            anal_data_dir = os.path.join(model_data_dir, '{}_{}_{}_{}_{}epoch_{}'.format(config['task_name'], 'ORD-train', config['fine_tune_from'], rxn_name, config['epochs'],current_time))
        else:
            anal_data_dir = os.path.join(model_data_dir, '{}_{}_{}_{}epoch_{}'.format(config['task_name'], config['fine_tune_from'], rxn_name, config['epochs'],current_time))
    else:
        anal_data_dir = os.path.join(model_data_dir, '{}_{}_{}epoch_{}'.format(config['task_name'], config['fine_tune_from'],config['epochs'],current_time))
    os.makedirs(anal_data_dir, exist_ok=True)
    if n != None:
        loss_df = pd.DataFrame({'epoch':epoch_list,'train_loss':train_loss_list,'valid_loss':valid_loss_list})
        loss_file = os.path.join(anal_data_dir, '{}_finetune_loss_data_{}.csv'.format(config['model_type'],n))
        loss_df.to_csv(loss_file, mode='w', index=False)
        p_l_file = os.path.join(anal_data_dir, '{}_finetune_pred_and_labels_{}'.format(config['model_type'],n))
        np.savez(p_l_file, pred=predictions,labels=labels)
    else:
        loss_df = pd.DataFrame({'epoch':epoch_list,'train_loss':train_loss_list,'valid_loss':valid_loss_list})
        loss_file = os.path.join(anal_data_dir, '{}_finetune_loss_data.csv'.format(config['model_type']))
        loss_df.to_csv(loss_file, mode='w', index=False)
        p_l_file = os.path.join(anal_data_dir, '{}_finetune_pred_and_labels'.format(config['model_type']))
        np.savez(p_l_file, pred=predictions,labels=labels)

        
def h_output(config, model_data_dir, h_set, label_set,n=None):
    import pickle
    h_dir = os.path.join(model_data_dir, 'h_data')
    os.makedirs(h_dir, exist_ok=True)
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        rxn_name = ''
        for i,rtype in enumerate(config['rxn_type']):
            if i > 0:
                rxn_name += '-'
            rxn_name += rtype
        h_dir_dir = os.path.join(h_dir, '{}_{}_{}_{}epoch_h_data'.format(config['task_name'],
                                                                         config['fine_tune_from'], 
                                                                         rxn_name, config['epochs']))
    else:
        h_dir_dir = os.path.join(h_dir, '{}_{}_{}epoch_h_data'.format(config['task_name'],
                                                                         config['fine_tune_from'], 
                                                                         config['epochs']))
    os.makedirs(h_dir_dir, exist_ok=True)
    if n != None:
        h_file = os.path.join(h_dir_dir, 'h_data_{}.pkl'.format(n))
        label_file = os.path.join(h_dir_dir, 'label_data_{}.pkl'.format(n))
    else:
        h_file = os.path.join(h_dir_dir, 'h_data.pkl')
        label_file = os.path.join(h_dir_dir, 'label_data.pkl')
    f1 = open(h_file,'wb')
    pickle.dump(h_set,f1)
    f1.close
    f2 = open(label_file,'wb')
    pickle.dump(label_set,f2)
    f2.close

def main(config):
    target_rxn=None
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        target_rxn= 'CN'
    if config['ORD_train']:
        ORD_path = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        train_val_smiles_list, train_val_labels = read_smiles_for_ORD(data_path=ORD_path, target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=None, target_rxn=None)
        train_val_smiles_set = [train_val_smiles_list, train_val_labels]
        _,_,test_smiles_list, test_labels = read_smiles_for_ORD(data_path=config['dataset']['data_path'], target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
        test_smiles_set = [test_smiles_list, test_labels]
        train_val_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=train_val_smiles_set,dataset_type='train_val')
        test_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=test_smiles_set,dataset_type='test')
        fine_tune = FineTune(train_val_dataset, config)
        epoch_list,train_loss_list,valid_loss_list,model = fine_tune.train()
        _,_,test_loader = test_dataset.get_data_loaders()
        predictions,labels = fine_tune._test(model, test_loader)
        save_calc_results(epoch_list,train_loss_list,valid_loss_list,predictions,labels) 
        if config['dataset']['task'] == 'classification':
            if config['task_name'] == 'PC':
                return fine_tune.accuracy
            else:
                return fine_tune.roc_auc
        if config['dataset']['task'] == 'regression':
            if config['task_name'] in ['qm7', 'qm8', 'qm9']:
                return fine_tune.mae
            else:
                return fine_tune.rmse
          
        
        
    elif config['CV']:
        if target_rxn != None:
            #read_scv -> smiles_list
            train_smiles_list, train_labels, target_smiles_list, target_labels = read_smiles_for_ORD(data_path=config['dataset']['data_path'],target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
            
        else:
            target_smiles_list, target_labels = read_smiles_for_ORD(data_path=config['dataset']['data_path'],target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
            train_smiles_list = []
            train_labels = []
        target_smiles_arry = np.array(target_smiles_list)
        target_labels_arry = np.array(target_labels)

        #smiles -> train_val/test_smiles (CV)
        target_data_idx = list(range(len(target_smiles_arry)))
        kf = KFold(n_splits=config['n_splits'],shuffle=True,random_state=0)
        result_score = 0
        for n,(train_val_idx,test_idx) in enumerate(kf.split(target_data_idx)): 
            train_val_smiles_list = copy.deepcopy(train_smiles_list)
            train_val_labels = copy.deepcopy(train_labels)
            #train_val_smiles_set, test_smiles_set 
            train_val_smiles_list.extend(target_smiles_arry[train_val_idx].tolist())
            train_val_labels.extend(target_labels_arry[train_val_idx].tolist())
            print('train_val_smiles: ', len(train_val_smiles_list))
            print('train_val_labels: ', len(train_val_labels))
            train_val_smiles_set = [train_val_smiles_list, train_val_labels]
            test_smiles_set = [target_smiles_arry[test_idx].tolist(), 
                               target_labels_arry[test_idx].tolist()]
            #train_val_dataset, test_dataset 
            train_val_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=train_val_smiles_set,dataset_type='train_val')
            test_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=test_smiles_set,dataset_type='test')
            fine_tune = FineTune(train_val_dataset, config)
            epoch_list,train_loss_list,valid_loss_list,model = fine_tune.train()
            _,_,test_loader = test_dataset.get_data_loaders()
            predictions,labels = fine_tune._test(model, test_loader)
            save_calc_results(epoch_list,train_loss_list,valid_loss_list,predictions,labels,n=n)
            if config['dataset']['task'] == 'classification':
                if config['task_name'] == 'PC':
                    result_score += fine_tune.accuracy
                else:
                    result_score += fine_tune.roc_auc
            if config['dataset']['task'] == 'regression':
                if config['task_name'] in ['qm7', 'qm8', 'qm9']:
                    result_score += fine_tune.mae
                else:
                    result_score += fine_tune.rmse
        return result_score/(n+1)
    else: 
        if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
            dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'])
        else:
            dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

        fine_tune = FineTune(dataset, config)
        epoch_list,train_loss_list,valid_loss_list,predictions,labels = fine_tune.train()
        save_calc_results(epoch_list,train_loss_list,valid_loss_list,predictions,labels) 
        if config['dataset']['task'] == 'classification':
            if config['task_name'] == 'PC':
                return fine_tune.accuracy
            else:
                return fine_tune.roc_auc
        if config['dataset']['task'] == 'regression':
            if config['task_name'] in ['qm7', 'qm8', 'qm9']:
                return fine_tune.mae
            else:
                return fine_tune.rmse
        
def h_main(config):
    from make_h_for_ORD import Make_h
    from models.SK_model_for_ORD import SK_model
    if config['model_type'] == 'gin':
        os.makedirs('GIN_data', exist_ok=True)
        model_data_dir = 'GIN_data'
    elif config['model_type'] == 'gcn':
        os.makedirs('GCN_data', exist_ok=True)
        model_data_dir = 'GCN_data'
    else:
        raise ValueError('Undefined model type!')
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        target_rxn= 'CN'
    if config['ORD_train']:
        ORD_path = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        train_val_smiles_list, train_val_labels = read_smiles_for_ORD(data_path=ORD_path, target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=None, target_rxn=None)
        train_val_smiles_set = [train_val_smiles_list, train_val_labels]
        _,_,test_smiles_list, test_labels = read_smiles_for_ORD(data_path=config['dataset']['data_path'], target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
        test_smiles_set = [test_smiles_list, test_labels]
        train_val_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=train_val_smiles_set,dataset_type='train_val')
        test_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=test_smiles_set,dataset_type='test')
        make_h = Make_h(train_val_dataset, config)
        train_h, train_h_labels, valid_h, valid_h_labels, model = make_h.pick_h()
        _,_,test_loader = test_dataset.get_data_loaders()
        test_h, test_h_labels = make_h._test(model, test_loader)
        h_set = {'train': train_h, 'valid': valid_h, 'test': test_h}
        label_set = {'train': train_h_labels, 'valid': valid_h_labels, 'test': test_h_labels}
        if config['output']:
            h_output(config, model_data_dir, h_set, label_set)
        if config['check_model'] != 'NONE':
            if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
                coment = 'ORD-train_'
            elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
                coment = 'ORD-train_'
                for i,t in enumerate(config['rxn_type']):
                    if i > 0:
                        coment += '-'
                    coment += t    
            sk_model = SK_model(h_set, label_set, config['dataset']['task'], config['check_model'],config['task_name'],config['fine_tune_from'],coment)
            if config['dataset']['task'] == 'classification':
                roc_auc, accuary = sk_model.calc()
                print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
                return roc_auc
            elif config['dataset']['task'] == 'regression':
                RMSE, R2 = sk_model.calc()
                print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
                return R2
        else:
            print('check_model was not selected. Calculation was finished')
            return 'NO calculation'

        
        
    elif config['CV']:
        if target_rxn != None:
            #read_scv -> smiles_list
            train_smiles_list, train_labels, target_smiles_list, target_labels = read_smiles_for_ORD(data_path=config['dataset']['data_path'],target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
            
        else:
            target_smiles_list, target_labels = read_smiles_for_ORD(data_path=config['dataset']['data_path'],target=config['dataset']['target'], task=config['dataset']['task'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
            train_smiles_list = []
            train_labels = []
        target_smiles_arry = np.array(target_smiles_list)
        target_labels_arry = np.array(target_labels)
        
        #smiles -> train_val/test_smiles (CV)
        target_data_idx = list(range(len(target_smiles_arry)))
        kf = KFold(n_splits=config['n_splits'],shuffle=True,random_state=0)
        result_score = 0
        start_time = datetime.now().strftime('%y%m%d%H%M')
        for n,(train_val_idx,test_idx) in enumerate(kf.split(target_data_idx)): 
            train_val_smiles_list = copy.deepcopy(train_smiles_list)
            train_val_labels = copy.deepcopy(train_labels)
            #train_val_smiles_set, test_smiles_set 
            train_val_smiles_list.extend(target_smiles_arry[train_val_idx].tolist())
            train_val_labels.extend(target_labels_arry[train_val_idx].tolist())
            print('train_val_smiles: ', len(train_val_smiles_list))
            print('train_val_labels: ', len(train_val_labels))
            train_val_smiles_set = [train_val_smiles_list, train_val_labels]
            test_smiles_set = [target_smiles_arry[test_idx].tolist(), 
                               target_labels_arry[test_idx].tolist()]
            #train_val_dataset, test_dataset 
            train_val_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=train_val_smiles_set,dataset_type='train_val')
            test_dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'],rxn_type=config['rxn_type'],smiles_set=test_smiles_set,dataset_type='test')
            
            make_h = Make_h(train_val_dataset, config)
            train_h, train_h_labels, valid_h, valid_h_labels, model = make_h.pick_h()
            _,_,test_loader = test_dataset.get_data_loaders()
            test_h, test_h_labels = make_h._test(model, test_loader)
            h_set = {'train': train_h, 'valid': valid_h, 'test': test_h}
            label_set = {'train': train_h_labels, 'valid': valid_h_labels, 'test': test_h_labels}   
            if config['output']:
                h_output(config, model_data_dir, h_set, label_set, n=n)
            if config['check_model'] != 'NONE':
                if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
                    coment = ''
                elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
                    coment = ''
                    for i,t in enumerate(config['rxn_type']):
                        if i > 0:
                            coment += '-'
                        coment += t
                sk_model = SK_model(h_set, label_set, config['dataset']['task'], config['check_model'],
                                    config['task_name'],config['fine_tune_from'],coment, n=n,
                                    start_time=start_time)
                if config['dataset']['task'] == 'classification':
                    roc_auc, accuary = sk_model.calc()
                    print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
                    result_score += roc_auc
                elif config['dataset']['task'] == 'regression':
                    RMSE, R2 = sk_model.calc()
                    print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
                    result_score += R2   
            else:
                print('check_model was not selected. Calculation was finished')
                result_score = 'NO calculation'
        if type(result_score) is str:
            return result_score
        else:
            return result_score/(n+1)

    else:
        if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
            dataset = MolTestDatasetWrapper(config['batch_size'],
                                            **config['dataset'],rxn_type=config['rxn_type'])
        else:
            dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        make_h = Make_h(dataset, config)
        h_set, label_set = make_h.pick_h()
        if config['output']:
            h_output(config, model_data_dir, h_set, label_set)
        if config['check_model'] != 'NONE':
            if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
                coment = ''
            elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
                coment = ''
                for i,t in enumerate(config['rxn_type']):
                    if i > 0:
                        coment += '-'
                    coment += t    
            sk_model = SK_model(h_set, label_set, config['dataset']['task'], config['check_model'],config['task_name'],config['fine_tune_from'],coment)
            if config['dataset']['task'] == 'classification':
                roc_auc, accuary = sk_model.calc()
                print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
                return roc_auc
            elif config['dataset']['task'] == 'regression':
                RMSE, R2 = sk_model.calc()
                print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
                return R2
        else:
            print('check_model was not selected. Calculation was finished')
            return 'NO calculation'

if __name__ == "__main__":
    config = yaml.load(open("config_finetune_for_ORD.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
        
    elif config['task_name'] == 'ORD':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        target_list = ['yield_round']
    elif config['task_name'] == 'ORD_rgr':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        target_list = ['yield']
    elif config['task_name'] == 'PC':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/PhotoCat/Photocatalyst_dataset.csv'
        target_list = ['yield_round']
    elif config['task_name'] == 'PC_rgr':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/PhotoCat/Photocatalyst_dataset.csv'
        target_list = ['yield']

    else:
        raise ValueError('Undefined downstream task!')
    current_time = datetime.now().strftime('%y%m%d_%H:%M:%S')
    print('calculation started at {}'.format(current_time))
    print(config)
    
    results_list = []
    if config['pick_h']:
        for target in target_list:
            config['dataset']['target'] = target
            result = h_main(config)
            results_list.append([target, result])
    else:
        for target in target_list:
            config['dataset']['target'] = target
            result = main(config)
            results_list.append([target, result])

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list)
    current_time = datetime.now().strftime('%y%m%d')
    if config['pick_h']:
        df.to_csv('experiments/{}_{}_{}_finetune_{}.csv'.format(config['check_model'], config['fine_tune_from'], config['task_name'],current_time), mode='a', index=False, header=False)
    else:
        df.to_csv('experiments/{}_{}_finetune_{}.csv'.format(config['fine_tune_from'], config['task_name'],current_time), mode='a', index=False, header=False)
    print('Calculation is finished')