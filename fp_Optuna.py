from fingerprint_for_ORD import read_csv, make_dataset

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import pickle
import warnings
warnings.simplefilter('ignore', FutureWarning) #FutureWarningの非表示

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold

import xgboost as xgb
import optuna

class Objective:
    def __init__(self, config, start_time):
        self.config = config 
        self.start_time = start_time
        
    def __call__(self, trial):
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
                                                               self.start_time)
                if self.config['calc']:
                    calc_dir += '{}_{}_{}_{}-fingerprint+calc_{}'.format(self.config['task_name'], 
                                                                self.config['model_type'],
                                                                self.config['fingerprint_type'],
                                                                rxn_names,
                                                                self.start_time)
            else:
                calc_dir = '{}_{}_{}-fingerprint_{}'.format(self.config['task_name'], 
                                                            self.config['model_type'],
                                                            self.config['fingerprint_type'], 
                                                            self.start_time)
        elif self.config['calc']:
            if self.config['task_name'] == 'PC' or self.config['task_name'] == 'PC_rgr':
                calc_dir = '{}_{}_{}_{}_only-calc_{}'.format(self.config['task_name'], 
                                                             self.config['model_type'], 
                                                             self.config['evaluation_function'],
                                                             rxn_names, self.start_time)
            else:
                calc_dir = '{}_{}_{}_only-calc_{}'.format(self.config['task_name'],
                                                          self.config['model_type'], 
                                                          self.config['evaluation_function'], 
                                                          self.start_time)
        if self.config['std']:
            calc_dir = 'std_-dEST' + calc_dir
        this_dir = os.path.join('Optuna_data', calc_dir)
        os.makedirs(this_dir, exist_ok=True)
        #trial number
        trial_number = trial.number

        #param
        if self.config['model_type'] == 'Lasso':
            params = {'alpha' : trial.suggest_uniform('alpha', 0.0, 1.0), 
                      'max_iter': int(trial.suggest_loguniform('max_iter', 100, 10000))}
        elif self.config['model_type'] == 'XGB':
            params = {'max_depth' : trial.suggest_int('max_depth', 3, 8, step=1), 
                      'min_child_weight': trial.suggest_int('min_child_weight', 1, 10, step=1), 
                      'gamma': trial.suggest_float('gamma', 0.0, 1.0, step=0.1), 
                      'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1), 
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                      'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10), 
                      'n_estimators': int(trial.suggest_loguniform('n_estimators', 10, 1000)), 
                      'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1.0), 
                      'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.0)}
        elif self.config['model_type'] == 'SVR':
            params = {'C': trial.suggest_loguniform('C', 1e-2, 1000),
                      'epsilon': trial.suggest_loguniform('epsilon', 1e-4, 1.0),
                      'gamma': trial.suggest_loguniform('gamma', 1e-7, 1000)}
        elif self.config['model_type'] == 'RFR':
            params = {'n_estimators': int(trial.suggest_loguniform('n_estimators', 10, 1000)),
                      'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                      'min_samples_split': int(trial.suggest_loguniform('min_samples_split', 2, 1000)), 
                      'max_depth': int(trial.suggest_loguniform('max_depth', 10, 1000))}
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
                model = Lasso(**params, random_state=0)
            elif self.config['model_type'] == 'XGB':
                model = xgb.XGBRegressor(**params, random_state=0)
            elif self.config['model_type'] == 'SVR':
                model = SVR(**params)
            elif self.config['model_type'] == 'RFR':
                model = RandomForestRegressor(**params, n_jobs=-1, random_state=0)
            
            #print(data_set['train'])
            #print(label_set['train'])
            model.fit(data_set['train'], label_set['train'])
            #modelの保存
            model_dir = os.path.join(this_dir,'model')
            os.makedirs(model_dir, exist_ok=True)
            model_filename = os.path.join(model_dir,'{}_model_{}-{}.pkl'.format(self.config['model_type'],
                                                                                trial_number,n))
            pickle.dump(model, open(model_filename, 'wb'))
            if self.config['model_type'] in ['Lasso', 'XGB', 'SVR', 'RFR']:
                pred = model.predict(data_set['test'])
            else:
                raise ValueError('Not defined model!')
            pred = np.array(pred)
            labels = np.array(label_set['test'])
            pred_list.append(pred)
            labels_list.append(labels)

        pred_sum = np.concatenate(pred_list)
        labels_sum = np.concatenate(labels_list)
        score = self.score_calculation(pred_sum, labels_sum)
        print('trial No.{} {} score: {:.3f}'.format(trial_number, self.config['evaluation_function'],
                                                    score))

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
            max_pred = np.argmax(pred, axis = 1)
            if self.config['evaluation_function'] == 'acu':
                score = accuracy_score(labels, max_pred)
            elif self.config['evaluation_function'] == 'roc':
                try:
                    score = roc_auc_score(one_hot_labels, pred, multi_class='ovr')
                except ValueError:
                    score = 'could not calculate'
            else:
                raise ValueError('Not suitable evaluation function!')
        elif self.config['dataset']['task'] == 'regression':
            if self.config['evaluation_function'] == 'RMSE':
                score = mean_squared_error(labels, pred, squared=False)
            elif self.config['evaluation_function'] == 'MSE':
                score = mean_absolute_error(labels, pred)
            elif self.config['evaluation_function'] == 'RMSLE':
                score = mean_squared_log_error(labels, pred, squared=False)
            elif self.config['evaluation_function'] == 'R2':
                score = r2_score(labels, pred)
            else:
                raise ValueError('Not suitable evaluation function!')
        return score


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
    #optuna.logging.enable_default_handler()#logの表示
    optuna.logging.disable_default_handler()#logを非表示
    TRIAL_SIZE = config['trial_size']
    objective = Objective(config, current_time)
    os.makedirs('study', exist_ok=True)
    study_name = os.path.join('study', '{}_{}_{}_study_{}'.format(config['task_name'], config['model_type'], "-".join(config['rxn_type']), current_time))
    if config['evaluation_function'] in ['acu', 'roc', 'R2']:
        study = optuna.create_study(study_name=study_name, 
                                    storage='sqlite:///'+study_name+".db",load_if_exists=True,
                                    direction='maximize')
    else:
        study = optuna.create_study(study_name=study_name, 
                                    storage='sqlite:///'+study_name+".db",load_if_exists=True,
                                    direction='minimize')
    study.optimize(objective, n_trials=TRIAL_SIZE)
    
    best_trial = study.best_trial
    best_num = study.best_trial.number
    best_value = study.best_value
    best_params = study.best_params
    
    bests = [best_num, best_value, best_params]
    
    print('best number:', best_num)
    print('best value:', best_value)
    print('best params:', best_params)
    print('best trial')
    print(best_trial)
    results_list = []
    best_names = ['best num', 'best value', 'best param']
    for b_name, best in zip(best_names,bests):
        results_list.append([b_name, best])
    os.makedirs('Optuna_results', exist_ok=True)
    df = pd.DataFrame(results_list)
    current_time = datetime.now().strftime('%y%m%d')
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        df.to_csv('Optuna_results/{}_{}_{}_{}-reults.csv'.format(config['task_name'],  config['model_type'], "-".join(config['rxn_type']), config['evaluation_function']), mode='a', index=False, header=False)
    else:
        df.to_csv('Optuna_results/{}_{}_{}-reults.csv'.format(config['task_name'],  config['model_type'], config['evaluation_function']), mode='a', index=False, header=False)

    print('Calculation is finished')
    