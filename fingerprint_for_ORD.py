import os
import csv
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

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*')  

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
import copy
from models.SK_model_for_ORD import SK_model

def read_csv(config,data_path, rxn_type, target_rxn=None):
    target = config['dataset']['target']
    task=config['dataset']['task']
    labels = []
    target_labels = []
    data_list = []
    target_data_list = []
    smiles_name_list = ['smiles0','smiles1','smiles2','product_smiles']
    calc_name_list = ['HOMO', 'LUMO', 'DM_g', 'E_S1', 'f_S1', 'E_T1', 'dEST']
    if target_rxn != None:
        if target_rxn not in rxn_type:
            raise ValueError('target_rxn must be incuded in rxn_type') 
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            rxn_set = []
            rxn_num = 0
            if rxn_type != None:
                if row['reaction_type'] not in rxn_type:
                    continue
            if config['fingerprint']:
                mol_list = []
                for name in smiles_name_list:
                    smiles = row[name]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol != None:
                        mol_list.append(mol)
                    else:
                        break
                if len(mol_list) < len(smiles_name_list): 
                    continue
                else:
                    for mol in  mol_list:
                        fp= list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
                        rxn_set.extend(fp)
            if config['calc']:
                for name in calc_name_list:
                    calc = row[name]
                    rxn_set.append(calc)
            label = row[target]
            if label != '':
                if target_rxn != None and row['reaction_type'] == target_rxn:
                    target_data_list.append(rxn_set)
                    if task == 'classification':
                        target_labels.append(int(float(label)//10)) #収率を1/10にして0~10の範囲に
                    elif task == 'regression':
                        target_labels.append(float(label)/10)
                    else:
                        raise ValueError('task must be either regression or classification') 
                else:
                    data_list.append(rxn_set)
                    if task == 'classification':
                        labels.append(int(float(label)//10)) #収率を1/10にして0~10の範囲に
                    elif task == 'regression':
                        labels.append(float(label)/10)
                    else:
                        raise ValueError('task must be either regression or classification')
    if target_rxn != None:
        print('target_data_list({}): '.format(target_rxn), len(target_data_list))
        print('data_list: ', len(data_list))
        print('target_labels({}): '.format(target_rxn), len(target_labels))
        print('labels: ', len(labels))
        return data_list, labels, target_data_list, target_labels
    else:
        print('data_list: ', len(data_list))
        print('labels: ', len(labels))
        return data_list, labels

def make_dataset(config, fp_set, data_type=None, test_data_set=None):
    if data_type == 'train_val':
        valid_size = config['dataset']['valid_size']
        test_size = 0
    else:
        valid_size = config['dataset']['valid_size']
        test_size = config['dataset']['test_size']
    # obtain training indices that will be used for validation
    fp_array = fp_set[0]
    y_data = fp_set[1]
    num_data = len(fp_array)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_data))
    split2 = int(np.floor(test_size * num_data))
    valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
    
    train_data = fp_array[train_idx]
    train_label = y_data[train_idx]
    valid_data = fp_array[valid_idx]
    valid_label = y_data[valid_idx]
    if data_type == 'train_val':
        test_data = test_data_set[0]
        test_label = test_data_set[1]
    else:
        test_data = fp_array[test_idx]
        test_label = y_data[test_idx]
    
    data_set = {'train': train_data, 'valid': valid_data, 'test': test_data}
    label_set = {'train': train_label, 'valid': valid_label, 'test': test_label}
    
    return data_set, label_set
    


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_fingerprint.yaml', os.path.join(model_checkpoints_folder, 'config_fingerprint.yaml'))

    
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

            
def main(config):
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        target_rxn= 'CN'
    else:
        target_rxn= None
    if config['ORD_train']:
        ORD_path = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        train_val_data_list, train_val_labels = read_csv(config, data_path=ORD_path, rxn_type=None, 
                                                           target_rxn=None)
        
        _,_,test_data_list, test_labels = read_csv(config, data_path=config['dataset']['data_path'],  
                                                     rxn_type=config['rxn_type'], target_rxn=target_rxn)
        train_val_array = np.array(train_val_data_list, dtype='float32')
        train_val_labels_array = np.array(train_val_labels, dtype='float32')
        print('train_val_data: ', len(train_val_data_array))
        print('train_val_labels: ', len(train_val_labels_array))
        train_val_data_set = [train_val_data_array, train_val_labels_array]
        test_data_set = [test_data_array, test_labels_array]
        data_set, label_set = make_dataset(config, train_val_data_set, data_type='train_val', 
                                           test_data_set=test_data_set)
        if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
            coment = 'ORD-train_'
        elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
            coment = 'ORD-train_'
            for i,t in enumerate(config['rxn_type']):
                if i > 0:
                    coment += '-'
                coment += t    
        sk_model = SK_model(data_set, label_set, config['dataset']['task'], config['model_type'],config['task_name'],None,coment)
        if config['dataset']['task'] == 'classification':
            roc_auc, accuary = sk_model.calc()
            print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
            return roc_auc
        elif config['dataset']['task'] == 'regression':
            RMSE, R2 = sk_model.calc()
            print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
            return R2
        
        
    elif config['CV']:
        if target_rxn != None:
            #read_scv -> smiles_list
            train_data_list, train_labels, target_data_list, target_labels = read_csv(config, data_path=config['dataset']['data_path'], rxn_type=config['rxn_type'], target_rxn=target_rxn)
            
        else:
            target_data_list, target_labels = read_csv(config, data_path=config['dataset']['data_path'], rxn_type=config['rxn_type'], target_rxn=None)
            train_data_list = []
            train_labels = []
        target_data_arry = np.array(target_data_list, dtype='float32')
        target_labels_arry = np.array(target_labels, dtype='float32')
        
        #smiles -> train_val/test_smiles (CV)
        target_data_idx = list(range(len(target_data_arry)))
        kf = KFold(n_splits=config['n_splits'],shuffle=True,random_state=0)
        result_score = 0
        start_time = datetime.now().strftime('%y%m%d%H%M')
        for n,(train_val_idx,test_idx) in enumerate(kf.split(target_data_idx)): 
            train_val_data_list = copy.deepcopy(train_data_list)
            train_val_labels = copy.deepcopy(train_labels)
            train_val_data_list.extend(target_data_arry[train_val_idx].tolist())
            train_val_labels.extend(target_labels_arry[train_val_idx].tolist())
            train_val_data_array = np.array(train_val_data_list, dtype='float32')
            train_val_labels_array = np.array(train_val_labels, dtype='float32')
            print('train_val_data: ', len(train_val_data_array))
            print('train_val_labels: ', len(train_val_labels_array))
            train_val_data_set = [train_val_data_array, train_val_labels_array]
            test_data_set = [target_data_arry[test_idx], 
                               target_labels_arry[test_idx]]
            data_set, label_set = make_dataset(config, train_val_data_set, data_type='train_val', 
                                               test_data_set=test_data_set)
            if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
                coment = ''
            elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
                if config['fingerprint']:
                    coment = 'only-fp_'
                    if config['calc']:
                        coment = 'fp+Calc_'
                elif config['calc']:
                    coment = 'only-calc_'
                else:    
                    coment = ''
                for i,t in enumerate(config['rxn_type']):
                    if i > 0:
                        coment += '-'
                    coment += t
            if config['fingerprint']:
                sk_model = SK_model(data_set, label_set, config['dataset']['task'], config['model_type'],
                                    config['task_name'],config['fingerprint_type'],coment, n=n,
                                    start_time=start_time)
            else:
                sk_model = SK_model(data_set, label_set, config['dataset']['task'], config['model_type'],
                                    config['task_name'],'',coment, n=n,
                                    start_time=start_time)
            if config['dataset']['task'] == 'classification':
                roc_auc, accuary = sk_model.calc()
                print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
                result_score += roc_auc
            elif config['dataset']['task'] == 'regression':
                RMSE, R2 = sk_model.calc()
                print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
                result_score += R2   

        return result_score/(n+1)

    else:
        if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
            target_data_list, target_labels = read_csv(config, data_path=config['dataset']['data_path'],
                                                        rxn_type=None, target_rxn=None)
        elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
            target_data_list, target_labels = read_csv(config, data_path=config['dataset']['data_path'], 
                                                       rxn_type=config['rxn_type'], target_rxn=None)
        for e in target_data_list:
            if len(e) != 8192:
                print(len(e))
                print(e)
        target_data_array = np.array(target_data_list, dtype='float32')
        target_labels_array = np.array(target_labels, dtype='float32')
        target_data_set = [target_data_array, target_labels_array]
        data_set, label_set = make_dataset(config, target_data_set)
        if config['task_name'] == 'ORD' or config['task_name'] == 'ORD_rgr':
            coment = ''
        elif config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
            if config['fingerprint']:
                coment = 'only-fp_'
                if config['calc']:
                    coment = 'fp+Calc_'
            elif config['calc']:
                coment = 'only-calc_'
            else:    
                coment = ''
            for i,t in enumerate(config['rxn_type']):
                if i > 0:
                    coment += '-'
                coment += t    
        if config['fingerprint']:
            sk_model = SK_model(data_set, label_set, config['dataset']['task'], config['model_type'],
                                    config['task_name'],config['fingerprint_type'],coment, n=n,
                                    start_time=start_time)
        else:
            sk_model = SK_model(data_set, label_set, config['dataset']['task'], config['model_type'],
                                    config['task_name'],'',coment, n=n,
                                    start_time=start_time)
        if config['dataset']['task'] == 'classification':
            roc_auc, accuary = sk_model.calc()
            print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
            return roc_auc
        elif config['dataset']['task'] == 'regression':
            RMSE, R2 = sk_model.calc()
            print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
            return R2
        

if __name__ == "__main__":
    config = yaml.load(open("config_fingerprint.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'ORD':
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
    for target in target_list:
        config['dataset']['target'] = target
        result = main(config)
        results_list.append([target, result])

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list)
    current_time = datetime.now().strftime('%y%m%d')
    df.to_csv('experiments/{}_{}_{}_fingerprint_{}.csv'.format(config['model_type'],  config['task_name'], config['fingerprint_type'], current_time), mode='a', index=False, header=False)
    print('Calculation is finished')