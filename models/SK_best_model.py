#RandomForest関連のimport
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,roc_auc_score,mean_squared_error,r2_score
#その他のimport
import numpy as np
import os
from datetime import datetime
import pickle
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class SK_best_model(object):
    def __init__(self, config, data_set, label_set, coment='', n=None, start_time=None):
        self.config = config
        self.train_data = data_set['train']
        self.valid_data = data_set['valid']
        self.test_data = data_set['test']
        self.coment = coment
        self.train_labels = label_set['train']
        self.valid_labels = label_set['valid']
        self.test_labels = label_set['test']
        self.n = n
        if start_time == None:
            start_time = datetime.now().strftime('%y%m%d%H%M')
        if self.config['model_type'] in ['RFR', 'Lasso', 'XGB','SVR']:
            data_dir = '{}_data'.format(self.config['model_type'])
            os.makedirs(data_dir, exist_ok=True)
            if self.config['std']:
                self.this_calc_dir = os.path.join(data_dir,
                                                  'std_-dEST_{}_{}_bestmodel_{}_{}'.format(self.config['task_name'], self.config['best']['evaluation_function'], self.coment, start_time))
            else:
                self.this_calc_dir = os.path.join(data_dir, '{}_{}_bestmodel_{}_{}'.format(self.config['task_name'], self.config['best']['evaluation_function'], self.coment, start_time))
            if self.config['train_test'] != True:
                os.makedirs(self.this_calc_dir, exist_ok=True)
        else:
            raise ValueError('Undefined model type!')
        
        if self.config['fingerprint']:
            if self.config['task_name'] == 'PC' or self.config['task_name'] == 'PC_rgr':
                calc_dir = '{}_{}_{}_{}-fingerprint_{}'.format(self.config['task_name'], 
                                                               self.config['model_type'], 
                                                               "-".join(config['rxn_type']), 
                                                              self.config['fingerprint_type'])
                if self.config['calc']:
                    calc_dir += '{}_{}_{}_{}-fingerprint+calc_{}'.format(self.config['task_name'], 
                                                                self.config['model_type'],
                                                                self.config['fingerprint_type'],
                                                                "-".join(config['rxn_type']))
            else:
                calc_dir = '{}_{}_{}-fingerprint_{}'.format(self.config['task_name'], 
                                                            self.config['model_type'],
                                                            self.config['fingerprint_type'])
        elif self.config['calc']:
            if self.config['task_name'] == 'PC' or self.config['task_name'] == 'PC_rgr':
                if self.config['std']:
                    calc_dir = 'std_-dEST{}_{}_{}_{}_only-calc_{}'.format(self.config['task_name'], 
                                                             self.config['model_type'], 
                                                        self.config['best']['evaluation_function'],
                                                             "-".join(config['rxn_type']),
                                                            self.config['best']['trial_date'])
                else:
                    calc_dir = '{}_{}_{}_{}_only-calc_{}'.format(self.config['task_name'], 
                                                             self.config['model_type'], 
                                                        self.config['best']['evaluation_function'],
                                                             "-".join(config['rxn_type']),
                                                            self.config['best']['trial_date'])
            else:
                calc_dir = '{}_{}_{}_only-calc_{}'.format(self.config['task_name'],
                                                          self.config['model_type'], 
                                                          self.config['best']['evaluation_function'])
        param_dir = os.path.join('Optuna_data', calc_dir,'params')
        param_file = 'trial_params_{}.pkl'.format(self.config['best']['trial_num'])
        param_path = os.path.join(param_dir, param_file)
        #self.params = pickle.load(param_path, 'rb')
        with open(param_path, 'rb') as f:
            self.params = pickle.load(f)
        
    def calc(self):
        if self.config['train_test']:
            train_pred, pred = self.best_calc_model()
            #評価
            train_labels = np.array(self.train_labels)
            labels = np.array(self.test_labels)
            if self.config['dataset']['task'] == 'classification':
                label_class = np.array(range(11))
                one_hot_train_labels = label_binarize(train_labels, classes=label_class)
                one_hot_labels = label_binarize(labels, classes=label_class)
                try:
                    train_roc_auc = roc_auc_score(one_hot_train_labels, train_pred, multi_class='ovr')
                    roc_auc = roc_auc_score(one_hot_labels, pred, multi_class='ovr')
                except ValueError:
                    train_roc_auc = 'could not calculate'
                    roc_auc = 'could not calculate'
                max_train_pred = np.argmax(train_pred, axis = 1)
                max_pred = np.argmax(pred, axis = 1)
                train_accuracy = accuracy_score(train_labels, max_train_pred)
                accuracy = accuracy_score(labels, max_pred)
                train_score = [train_roc_auc, train_accuracy]
                test_score = [roc_auc, accuracy]
                return train_score, test_score
            elif self.config['dataset']['task'] == 'regression':
                train_RMSE = mean_squared_error(train_labels, train_pred, squared=False)
                train_R2 = r2_score(train_labels, train_pred)
                RMSE = mean_squared_error(labels, pred, squared=False)
                R2 = r2_score(labels, pred)
                train_score = [train_RMSE, train_R2]
                test_score = [RMSE, R2]
                return train_score, test_score
        else:
            pred = self.best_calc_model()
            #評価
            labels = np.array(self.test_labels)
            #予測とラベルの保存
            if self.n != None:
                PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL_{}'.format(self.config['model_type'],self.n))
            else:
                PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL'.format(self.config['model_type']))
            np.savez(PandL_filename, pred=pred,labels=labels)
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
        
    def best_calc_model(self):
        print('best {} calculation is started!'.format(self.config['model_type']))
        if self.config['model_type'] == 'RFC':
            model = RandomForestClassifier(n_jobs=-1, random_state=0)
        elif self.config['model_type'] == 'LR':
            model = LogisticRegression(max_iter=10000,random_state=0)
        elif self.config['model_type'] == 'RFR':
            self.params['n_estimators'] = int(self.params['n_estimators'])
            self.params['min_samples_split'] = int(self.params['min_samples_split'])
            self.params['max_depth'] = int(self.params['max_depth'])
            model = RandomForestRegressor(**self.params, n_jobs=-1, random_state=0)
        elif self.config['model_type'] == 'Lasso':
            self.params['max_iter'] = int(self.params['max_iter'])
            model = Lasso(**self.params, random_state=0)
        elif self.config['model_type'] == 'XGB':
            self.params['n_estimators'] = int(self.params['n_estimators'])
            model = xgb.XGBRegressor(**self.params, random_state=0)
        elif self.config['model_type'] == 'SVR':
            model = SVR(**self.params)
        else:
            raise ValueError('Undefined model type!')
        model.fit(self.train_data, self.train_labels)
        if self.config['train_test'] != True:
            #modelの保存
            if self.n != None:
                model_filename = os.path.join(self.this_calc_dir,'{}_model_{}.pkl'.format(self.config['model_type'], self.n))
                pickle.dump(model, open(model_filename, 'wb'))
            else:
                model_filename = os.path.join(self.this_calc_dir,'{}_model.pkl'.format(self.config['model_type']))
                pickle.dump(model, open(model_filename, 'wb'))
        #予測値
        if self.config['train_test']:
            if self.config['model_type'] in ['RFC', 'LR']:
                train_pred = model.predict_proba(self.train_data)
                pred = model.predict_proba(self.test_data)
            else:
                train_pred = model.predict(self.train_data)
                pred = model.predict(self.test_data)
            train_pred = np.array(train_pred)
            pred = np.array(pred)
            return train_pred, pred
        else:
            if self.config['model_type'] in ['RFC', 'LR']:
                pred = model.predict_proba(self.test_data)
            else:
                pred = model.predict(self.test_data)
            pred = np.array(pred)
            return pred
        
'''
    def calc(self):
        if self.config['model_type'] == 'RFR':
            pred = self.best_RF_Regressor()
        elif self.config['model_type'] == 'Lasso':
            pred = self.best_Lasso_rgr()
        elif self.config['model_type'] == 'XGB':
            pred = self.best_xgb_rgr()
        elif self.config['model_type'] == 'SVR':
            pred = self.best_sv_rgr()
        else:
            raise ValueError('Undefined model type!')
        #評価
        labels = np.array(self.test_labels)
        #予測とラベルの保存
        if self.n != None:
            PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL_{}'.format(self.config['model_type'],self.n))
        else:
            PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL'.format(self.config['model_type']))
        np.savez(PandL_filename, pred=pred,labels=labels)
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
        
    def logistic_regression(self):
        print('LR calculation is started!')
        lr = LogisticRegression(max_iter=10000,random_state=0)
        lr.fit(self.train_data, self.train_labels)
        #modelの保存
        if self.n != None:
            model_filename = os.path.join(self.this_calc_dir,'LR_model_{}.pkl'.format(self.n))
            pickle.dump(lr, open(model_filename, 'wb'))
        else:
            model_filename = os.path.join(self.this_calc_dir,'LR_model.pkl')
            pickle.dump(lr, open(model_filename, 'wb'))
        #予測値
        pred = lr.predict_proba(self.test_data)
        pred = np.array(pred)
        return pred
        
    
    def random_forest(self):
        print('RFC calculation is started!')
        rfc = RandomForestClassifier(n_jobs=-1, random_state=0)
        rfc.fit(self.train_data, self.train_labels)
        #modelの保存
        if self.n != None:
            model_filename = os.path.join(self.this_calc_dir,'RFC_model_{}.pkl'.format(self.n))
            pickle.dump(rfc, open(model_filename, 'wb'))
        else:
            model_filename = os.path.join(self.this_calc_dir,'RFC_model.pkl')
            pickle.dump(rfc, open(model_filename, 'wb'))
        #予測値
        pred = rfc.predict_proba(self.test_data)
        pred = np.array(pred)
        return pred
        
    def best_RF_Regressor(self):
        print('best RFR calclation is started!')
        self.params['n_estimators'] = int(self.params['n_estimators'])
        self.params['min_samples_split'] = int(self.params['min_samples_split'])
        self.params['max_depth'] = int(self.params['max_depth'])
        rfr = RandomForestRegressor(**self.params, n_jobs=-1, random_state=0)
        rfr.fit(self.train_data, self.train_labels)
        #modelの保存
        if self.n != None:
            model_filename = os.path.join(self.this_calc_dir,'RFR_model_{}.pkl'.format(self.n))
            pickle.dump(rfr, open(model_filename, 'wb'))
        else:
            model_filename = os.path.join(self.this_calc_dir,'RFR_model.pkl')
            pickle.dump(rfr, open(model_filename, 'wb'))
        #予測値
        pred = rfr.predict(self.test_data)
        pred = np.array(pred)
        return pred
    
    def best_Lasso_rgr(self):
        print('Lasso calclation is started!')
        #self.params['max_iter'] = 10000
        self.params['max_iter'] = int(self.params['max_iter'])
        lasso = Lasso(**self.params, random_state=0)
        lasso.fit(self.train_data, self.train_labels)
        #modelの保存
        if self.n != None:
            model_filename = os.path.join(self.this_calc_dir,'Lasso_model_{}.pkl'.format(self.n))
            pickle.dump(lasso, open(model_filename, 'wb'))
        else:
            model_filename = os.path.join(self.this_calc_dir,'Lasso_model.pkl')
            pickle.dump(lasso, open(model_filename, 'wb'))
        #予測値
        pred = lasso.predict(self.test_data)
        pred = np.array(pred)
        return pred
    
    def best_xgb_rgr(self):
        print('XGBoost calclation is started!')
        self.params['n_estimators'] = int(self.params['n_estimators'])
        xgbr = xgb.XGBRegressor(**self.params, random_state=0)
        xgbr.fit(self.train_data, self.train_labels)
        #modelの保存
        if self.n != None:
            model_filename = os.path.join(self.this_calc_dir,'XGB_model_{}.pkl'.format(self.n))
            pickle.dump(xgbr, open(model_filename, 'wb'))
        else:
            model_filename = os.path.join(self.this_calc_dir,'XGB_model.pkl')
            pickle.dump(xgbr, open(model_filename, 'wb'))
        #予測値
        pred = xgbr.predict(self.test_data)
        pred = np.array(pred)
        return pred
    
    def best_sv_rgr(self):
        print('SVR calclation is started!')
        svr = SVR(**self.params)
        svr.fit(self.train_data, self.train_labels)
        #modelの保存
        if self.n != None:
            model_filename = os.path.join(self.this_calc_dir,'SVR_model_{}.pkl'.format(self.n))
            pickle.dump(svr, open(model_filename, 'wb'))
        else:
            model_filename = os.path.join(self.this_calc_dir,'SVR_model.pkl')
            pickle.dump(svr, open(model_filename, 'wb'))
        #予測値
        pred = svr.predict(self.test_data)
        pred = np.array(pred)
        return pred
'''