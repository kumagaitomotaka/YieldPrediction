from fingerprint_for_ORD import read_csv, make_dataset

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error,r2_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold

import optuna

class Objective:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, trial):
        start_time = datetime.now().strftime('%y%m%d%H%M')
        os.makedirs('Optuna_data', exist_ok=True)
        if self.config['task_name'] == 'PC' or self.config['task_name'] == 'PC_rgr': 
            target_rxn= 'CN'
            rxn_names = ''
            for n,r in enumerate(self.config['rxn_type']):
                if n > 0: rxn_names += '-'
                rxn_names += r 
        else:
            target_rxn= None
        if self.config['fingerprint']:
            if self.config['task_name'] == 'PC' or self.config['task_name'] == 'PC_rgr':
                calc_dir = '{}_{}_{}_{}-fingerprint_{}'.format(self.config['task_name'], 
                                                               self.config['model_type'], 
                                                               rxn_names, self.config['fingerprint_type'],
                                                               start_time)
            else:
                calc_dir = '{}_{}_{}-fingerprint_{}'.format(self.config['task_name'], 
                                                            self.config['model_type'],
                                                            self.config['fingerprint_type'], start_time)
            if self.config['calc']:
                calc_dir += '+calc'
        elif self.config['calc']:
            if self.config['task_name'] == 'PC' or self.config['task_name'] == 'PC_rgr':
                calc_dir = '{}_{}_{}_only-calc_{}'.format(self.config['task_name'], 
                                                          self.config['model_type'],rxn_names, start_time)
            else:
                calc_dir = '{}_{}_only-calc_{}'.format(self.config['task_name'], self.config['model_type'], start_time)
        os.makedirs(calc_dir, exist_ok=True)
        this_dir = os.path.join('Optuna_data', calc_dir)
        #trial number
        trial_number = trial.number

        #param
        if self.config['model_type'] == 'Lasso':
            params = {'alpha' : trial.suggest_uniform('alpha', 0.1, 1.0)}
        #パラメータ
        trial_params = trial.params

        #dataset
        if target_rxn != None:
            #read_scv -> smiles_list
            train_data_list, train_labels, target_data_list, target_labels = read_csv(self.config, data_path=self.config['dataset']['data_path'], rxn_type=self.config['rxn_type'], target_rxn=target_rxn)   
        else:
            target_data_list, target_labels = read_csv(self.config, data_path=self.config['dataset']['data_path'], rxn_type = self.config['rxn_type'], target_rxn=None)
            train_data_list = []
            train_labels = []
        target_data_arry = np.array(target_data_list, dtype='float32')
        target_labels_arry = np.array(target_labels, dtype='float32')

        #smiles -> train_val/test_smiles (CV)
        target_data_idx = list(range(len(target_data_arry)))
        kf = KFold(n_splits=self.config['n_splits'],shuffle=True,random_state=0)
        result_score = 0
        start_time = datetime.now().strftime('%y%m%d%H%M')
        pred_list = []
        labels_list = []
        for n,(train_val_idx,test_idx) in enumerate(kf.split(target_data_idx)): 
            train_val_data_list = copy.deepcopy(train_data_list)
            train_val_labels = copy.deepcopy(train_labels)
            train_val_data_list.extend(target_data_arry[train_val_idx].tolist())
            train_val_labels.extend(target_labels_arry[train_val_idx].tolist())
            train_val_data_array = np.array(train_val_data_list, dtype='float32')
            train_val_labels_array = np.array(train_val_labels, dtype='float32')
            train_val_data_set = [train_val_data_array, train_val_labels_array]
            test_data_set = [target_data_arry[test_idx], target_labels_arry[test_idx]]
            data_set, label_set = make_dataset(self.config, train_val_data_set, data_type='train_val',
                                               test_data_set=test_data_set)

            #model
            if self.config['model_type'] == 'Lasso':
                model = Lasso(**params, max_iter=10000, random_state=0)

            model.fit(data_set['train'],label_set['train'])
            #modelの保存
            model_dir = os.path.join(this_dir,'model')
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir,'{}_model_{}-{}.pkl'.format(self.config['model_type'],
                                                                                trial_number,n))
            pickle.dump(model, open(model_filename, 'wb'))
            if self.config['model_type'] == 'Lasso':
                pred = model.predict(data_set['test'])
            else:
                raise ValueError('Not defined model!')
            pred = np.array(pred)
            labels = np.array(label_set['test'])
            pred_list.append(pred)
            labels_list.append(labels)

        pred_sum = np.concatenate(pred_list)
        labels_sum = np.concatenate(labels_list)
        score1, score2 = self.score_calculation(pred_sum, labels_sum) #score1: roc_auc or RMSE 
                                                                         #score2: accuracy or R2
        if self.config['dataset']['task'] == 'classification':
            score = score1
            print('trial No.{} ROC score: {:.3f}'.format(trial_number, score))
        elif self.config['dataset']['task'] == 'regression':
            score = score2
            print('trial No.{} R2 score: {:.3f}'.format(trial_number, score))

        #trialの保存
        param_dir = os.path.join(this_dir, 'params')
        os.makedirs(param_dir, exist_ok=True)
        trial_path = os.path.join(param_dir, 'trial_params_{}.pkl'.format(trial_number))
        pickle.dump(trial_params, open(trial_path, 'wb'))

        return score

    def score_calculation(self, pred, labels):
        if self.config['dataset']['task'] == 'classification':
            label_class = np.array(range(11))
            one_hot_labels = label_binarize(labels, classes=label_class)
            try:
                roc_auc = roc_auc_score(one_hot_labels, pred, multi_class='ovr')
            except ValueError:
                roc_auc = 'could not calculate'
            max_pred = np.argmax(pred, axis = 1)
            accuracy = accuracy_score(labels, max_pred)
            return roc_auc, accuracy
        elif self.config['dataset']['task'] == 'regression':
            RMSE = mean_squared_error(labels, pred, squared=False)
            R2 = r2_score(labels, pred)
            return RMSE, R2


if __name__ == "__main__":
    config = yaml.load(open("config_for_fp-Optuna.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'ORD':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        config['dataset']['target'] = 'yield_round'
    elif config['task_name'] == 'ORD_rgr':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        config['dataset']['target'] = 'yield'
    elif config['task_name'] == 'PC':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/PhotoCat/Photocatalyst_dataset.csv'
        config['dataset']['target'] = 'yield_round'
    elif config['task_name'] == 'PC_rgr':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/PhotoCat/Photocatalyst_dataset.csv'
        config['dataset']['target'] = 'yield'

    else:
        raise ValueError('Undefined downstream task!')
    current_time = datetime.now().strftime('%y%m%d%H%M')
    print('calculation started at {}'.format(current_time))
    print(config)
    #↓↓
    #パラメータチューニングの実行
    optuna.logging.enable_default_handler()#logの表示
    TRIAL_SIZE = 10
    objective = Objective(config)
    study_name = '{}_{}_study_{}'.format(config['task_name'], config['model_type'], current_time)
    study = optuna.create_study(study_name=study_name, 
                                storage='sqlite:///'+study_name+".db",load_if_exists=True,
                                direction='maximize')
    study.optimize(objective, n_trials=TRIAL_SIZE)
    
    best_trial = study.best_trial
    best_value = study.best_value
    
    bests = [best_trial,best_value]
    
    print('best value:', best_value)
    print('best trial')
    print(best_trial)
    results_list = []
    best_names = ['best_trial', 'best_value']
    for b_name, best in zip(best_names,bests):
        results_list.append([b_name, best])
    
    os.makedirs('Optuna_results', exist_ok=True)
    df = pd.DataFrame(results_list)
    current_time = datetime.now().strftime('%y%m%d')
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        df.to_csv('Optuna_results/{}_{}_{}_reults.csv'.format(config['task_name'],  config['model_type'], "-".join(config['rxn_type'])), mode='a', index=False, header=False)
    else:
        df.to_csv('Optuna_results/{}_{}_reults.csv'.format(config['task_name'],  config['model_type']), mode='a', index=False, header=False)
    print('Calculation is finished')