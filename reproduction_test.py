import os
import csv
import shutil
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import sys

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*')  

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
import copy
from models.SK_model_for_ORD import SK_model
from models.SK_best_model import SK_best_model
from fingerprint_for_ORD import read_csv, make_dataset

def main(config, seed):
    seed = seed
    if config['task_name'] == 'PC' or config['task_name'] == 'PC_rgr': 
        target_rxn= 'CN'
    else:
        target_rxn= None
    if config['CV']:
        train_results = []
        results = []
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
        kf = KFold(n_splits=config['n_splits'],shuffle=True,random_state=seed)
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
            if config['data_pick_up']:
                pickup_data_dir = 'data/PhotoCat/sampling'
                os.makedirs(pickup_data_dir, exist_ok=True)
                pickup_train = os.path.join(pickup_data_dir, 'train_data_idx_{}-{}'.format(seed,n))
                pickup_test = os.path.join(pickup_data_dir, 'test_data_idx_{}-{}'.format(seed,n))
                pickle.dump(train_val_idx, open(pickup_train, 'wb'))
                pickle.dump(test_idx, open(pickup_test, 'wb'))
                continue
            if config['best_model']:
                    sk_model = SK_best_model(config, data_set, label_set, coment=coment, n=n,
                                             start_time=start_time)
            elif config['fingerprint']:
                sk_model = SK_model(data_set, label_set, config, 
                                    config['fingerprint_type'],coment, n=n, start_time=start_time, 
                                    train_test=config['train_test'])
            else:
                sk_model = SK_model(data_set, label_set, config,'',coment, n=n,
                                    start_time=start_time, train_test=config['train_test'])
            if config['dataset']['task'] == 'classification':
                if config['train_test']:
                    train_score, test_score = sk_model.calc()
                    print('test ROC-AUC score: {}, accuarcy: {}'.format(test_score[0], test_score[1]))
                    train_results.append(train_score[0])
                    results.append(test_score[0])
                else:
                    roc_auc, accuary = sk_model.calc()
                    print('ROC-AUC score: {}, accuarcy: {}'.format(roc_auc, accuary))
                    results.appned(roc_auc)
            elif config['dataset']['task'] == 'regression':
                if config['train_test']:
                    train_score, test_score = sk_model.calc()
                    print('test RMSE: {}, R2 value: {}'.format(test_score[0], test_score[1]))
                    train_results.append(train_score[1])
                    results.append(test_score[1])
                else:
                    RMSE, R2 = sk_model.calc()
                    print('RMSE: {}, R2 value: {}'.format(RMSE, R2))
                    results.append(R2)   

        return train_results, results

    else:
        raise ValueError('Only CV is defined!')
        

if __name__ == "__main__":
    config = yaml.load(open("config_reproduction_test.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'ORD':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        target = 'yield_round'
    elif config['task_name'] == 'ORD_rgr':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/ORD/ORD_dataset_for_MolCLR.csv'
        target = 'yield'
    elif config['task_name'] == 'PC':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/PhotoCat/Photocatalyst_dataset.csv'
        target = 'yield_round'
    elif config['task_name'] == 'PC_rgr':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/PhotoCat/Photocatalyst_dataset.csv'
        target = 'yield'

    else:
        raise ValueError('Undefined downstream task!')
    current_time = datetime.now().strftime('%y%m%d_%H:%M:%S')
    print('calculation started at {}'.format(current_time))
    print(config)
    
    try_num = config['try_num']
    results_list = []
    config['dataset']['target'] = target
    for seed in range(try_num):
        if config['data_pick_up']:
            main(config, seed)
        elif config['train_test']:
            train_results, results = main(config, seed)
            for i in range(config['n_splits']):
                results_list.append([seed, i, train_results[i], results[i]])
        else:   
            _, result = main(config, seed)
            for i in range(config['n_splits']):
                results_list.append([seed, i, results[i]])
    
    if config['data_pick_up']: sys.exit()
    os.makedirs('experiments', exist_ok=True)
    repro_dir = os.path.join('experiments', 'reproductions')
    os.makedirs(repro_dir, exist_ok=True)
    if config['train_test']:
        df = pd.DataFrame(results_list, columns = ['seed', 'CV', 'train', 'test'])
    else:
        df = pd.DataFrame(results_list, columns = ['seed', 'CV', 'test'])
    current_time = datetime.now().strftime('%y%m%d')
    if config['std']:
        csv_name = os.path.join(repro_dir, '{}_{}_{}_std_HL_gap_{}.csv'.format(config['task_name'],  config['model_type'], "-".join(config['rxn_type']), current_time))
    else:
        csv_name = os.path.join(repro_dir, '{}_{}_{}_{}.csv'.format(config['task_name'],  config['model_type'], "-".join(config['rxn_type']), current_time))
    df.to_csv(csv_name, mode='a', index=False, header=True)
    print('Calculation is finished')