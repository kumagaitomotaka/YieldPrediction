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

class SK_model(object):
    def __init__(self, data_set, label_set, config, fine_tune_form = 'pretraind_gin',coment = '',n=None,start_time=None, train_test=False):
        self.train_data = data_set['train']
        self.valid_data = data_set['valid']
        self.test_data = data_set['test']
        self.train_labels = label_set['train']
        self.valid_labels = label_set['valid']
        self.test_labels = label_set['test']
        self.config = config
        self.task = config['dataset']['task']
        self.model_type = config['model_type']
        self.task_name = config['task_name']
        self.fine_tune_form = fine_tune_form
        self.coment = coment
        self.n = n
        self.train_test = train_test
        if start_time == None:
            start_time = datetime.now().strftime('%y%m%d%H%M')
        if train_test != True:
            if self.model_type == 'RFC':
                os.makedirs('RFC_data', exist_ok=True)
                self.this_calc_dir = 'RFC_data/{}_{}_{}_{}'.format(self.task_name, self.fine_tune_form, self.coment, start_time)
                os.makedirs(self.this_calc_dir)
            elif self.model_type == 'LR':
                os.makedirs('LR_data', exist_ok=True)
                self.this_calc_dir = 'LR_data/{}_{}_{}_{}'.format(self.task_name, self.fine_tune_form, self.coment, start_time)
                os.makedirs(self.this_calc_dir)
            elif self.model_type == 'RFR':
                os.makedirs('RFR_data', exist_ok=True)
                self.this_calc_dir = 'RFR_data/{}_{}_{}_{}'.format(self.task_name, self.fine_tune_form, self.coment, start_time)
                os.makedirs(self.this_calc_dir, exist_ok=True)
            elif self.model_type == 'Lasso':
                os.makedirs('Lasso_data', exist_ok=True)
                self.this_calc_dir = 'Lasso_data/{}_{}_{}_{}'.format(self.task_name, self.fine_tune_form, self.coment, start_time)
                os.makedirs(self.this_calc_dir, exist_ok=True)
            elif self.model_type == 'XGB':
                os.makedirs('XGB_data', exist_ok=True)
                self.this_calc_dir = 'XGB_data/{}_{}_{}_{}'.format(self.task_name, self.fine_tune_form, self.coment, start_time)
                os.makedirs(self.this_calc_dir, exist_ok=True)
            elif self.model_type == 'SVR':
                os.makedirs('SVR_data', exist_ok=True)
                self.this_calc_dir = 'SVR_data/{}_{}_{}_{}'.format(self.task_name, self.fine_tune_form, self.coment, start_time)
                os.makedirs(self.this_calc_dir, exist_ok=True)
            else:
                raise ValueError('Undefined model type!')
        
    def calc(self):
        if self.train_test:
            train_pred, pred = self.calc_model()
            #評価
            train_labels = np.array(self.train_labels)
            labels = np.array(self.test_labels)
            if self.task == 'classification':
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
            elif self.task == 'regression':
                train_RMSE = mean_squared_error(train_labels, train_pred, squared=False)
                train_R2 = r2_score(train_labels, train_pred)
                RMSE = mean_squared_error(labels, pred, squared=False)
                R2 = r2_score(labels, pred)
                train_score = [train_RMSE, train_R2]
                test_score = [RMSE, R2]
                return train_score, test_score
        else:
            pred = self.calc_model()
            #評価
            labels = np.array(self.test_labels)
            #予測とラベルの保存
            if self.n != None:
                PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL_{}'.format(self.model_type,self.n))
            else:
                PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL'.format(self.model_type))
            np.savez(PandL_filename, pred=pred,labels=labels)
            if self.task == 'classification':
                label_class = np.array(range(11))
                one_hot_labels = label_binarize(labels, classes=label_class)
                try:
                    roc_auc = roc_auc_score(one_hot_labels, pred, multi_class='ovr')
                except ValueError:
                    roc_auc = 'could not calculate'
                max_pred = np.argmax(pred, axis = 1)
                accuracy = accuracy_score(labels, max_pred)
                return roc_auc, accuracy
            elif self.task == 'regression':
                RMSE = mean_squared_error(labels, pred, squared=False)
                R2 = r2_score(labels, pred)
                return RMSE, R2
        
    def calc_model(self):
        print('{} calculation is started!'.format(self.model_type))
        if self.model_type == 'RFC':
            model = RandomForestClassifier(n_jobs=-1, random_state=0)
        elif self.model_type == 'LR':
            model = LogisticRegression(max_iter=10000,random_state=0)
        elif self.model_type == 'RFR':
            model = RandomForestRegressor(n_jobs=-1, random_state=0)
        elif self.model_type == 'Lasso':
            model = Lasso(max_iter=10000,random_state=0)
        elif self.model_type == 'XGB':
            model = xgb.XGBRegressor(max_depth=3,n_jobs=-1, random_state=0)
        elif self.model_type == 'SVR':
            model = SVR()
        else:
            raise ValueError('Undefined model type!')
        model.fit(self.train_data, self.train_labels)
        if self.train_test != True:
            #modelの保存
            if self.n != None:
                model_filename = os.path.join(self.this_calc_dir,'{}_model_{}.pkl'.format(self.model_type, self.n))
                pickle.dump(model, open(model_filename, 'wb'))
            else:
                model_filename = os.path.join(self.this_calc_dir,'{}_model.pkl'.format(self.model_type))
                pickle.dump(model, open(model_filename, 'wb'))
        #予測値
        if self.train_test:
            if self.model_type in ['RFC', 'LR']:
                train_pred = model.predict_proba(self.train_data)
                pred = model.predict_proba(self.test_data)
            else:
                train_pred = model.predict(self.train_data)
                pred = model.predict(self.test_data)
            train_pred = np.array(train_pred)
            pred = np.array(pred)
            return train_pred, pred
        else:
            if self.model_type in ['RFC', 'LR']:
                pred = model.predict_proba(self.test_data)
            else:
                pred = model.predict(self.test_data)
            pred = np.array(pred)
            return pred
'''
    def calc(self):
        if self.model_type == 'RFC':
            pred = self.random_forest()
        elif self.model_type == 'LR':
            pred = self.logistic_regression()
        elif self.model_type == 'RFR':
            pred = self.RF_Regressor()
        elif self.model_type == 'Lasso':
            pred = self.Lasso_rgr()
        elif self.model_type == 'XGB':
            pred = self.xgb_rgr()
        elif self.model_type == 'SVR':
            pred = self.sv_rgr()
        else:
            raise ValueError('Undefined model type!')
        #評価
        labels = np.array(self.test_labels)
        #予測とラベルの保存
        if self.n != None:
            PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL_{}'.format(self.model_type,self.n))
        else:
            PandL_filename = os.path.join(self.this_calc_dir,'{}_PandL'.format(self.model_type))
        np.savez(PandL_filename, pred=pred,labels=labels)
        if self.task == 'classification':
            label_class = np.array(range(11))
            one_hot_labels = label_binarize(labels, classes=label_class)
            try:
                roc_auc = roc_auc_score(one_hot_labels, pred, multi_class='ovr')
            except ValueError:
                roc_auc = 'could not calculate'
            max_pred = np.argmax(pred, axis = 1)
            accuracy = accuracy_score(labels, max_pred)
            return roc_auc, accuracy
        elif self.task == 'regression':
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
    
    def RF_Regressor(self):
        print('RFR calclation is started!')
        rfr = RandomForestRegressor(n_jobs=-1, random_state=0)
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
    
    def Lasso_rgr(self):
        print('Lasso calclation is started!')
        lasso = Lasso(max_iter=10000,random_state=0)
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
    
    def xgb_rgr(self):
        print('XGBoost calclation is started!')
        xgbr = xgb.XGBRegressor(max_depth=3,n_jobs=-1, random_state=0)
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
    
    def sv_rgr(self):
        print('SVR calclation is started!')
        svr = SVR()
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
    