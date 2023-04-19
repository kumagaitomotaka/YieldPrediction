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

class SK_model(object):
    def __init__(self, data_set, label_set, task='classification', model_type='RFC',task_name='ORD', fine_tune_form = 'pretraind_gin',coment = '',n=None,start_time=None):
        self.train_data = data_set['train']
        self.valid_data = data_set['valid']
        self.test_data = data_set['test']
        self.train_labels = label_set['train']
        self.valid_labels = label_set['valid']
        self.test_labels = label_set['test']
        self.task = task
        self.model_type = model_type
        self.task_name = task_name
        self.fine_tune_form = fine_tune_form
        self.coment = coment
        self.n = n
        if start_time == None:
            start_time = datetime.now().strftime('%y%m%d%H%M')
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
        else:
            raise ValueError('Undefined model type!')
        
    
    def calc(self):
        if self.model_type == 'RFC':
            pred = self.random_forest()
        elif self.model_type == 'LR':
            pred = self.logistic_regression()
        elif self.model_type == 'RFR':
            pred = self.RF_Regressor()
        elif self.model_type == 'Lasso':
            pred = self.Lasso_rgr()
        else:
            raise ValueError('Undefined model type!')
        #評価
        labels = np.array(self.test_labels)
        #予測とラベルの保存
        if self.n != None:
            if self.model_type == 'RFC':
                PandL_filename = os.path.join(self.this_calc_dir,'RFC_PandL_{}'.format(self.n))
            elif self.model_type == 'LR':
                PandL_filename = os.path.join(self.this_calc_dir,'LR_PandL_{}'.format(self.n))
            elif self.model_type == 'RFR':
                PandL_filename = os.path.join(self.this_calc_dir,'RFR_PandL_{}'.format(self.n))
            elif self.model_type == 'Lasso':
                PandL_filename = os.path.join(self.this_calc_dir,'Lasso_PandL_{}'.format(self.n))
        else:
            if self.model_type == 'RFC':
                PandL_filename = os.path.join(self.this_calc_dir,'RFC_PandL')
            elif self.model_type == 'LR':
                PandL_filename = os.path.join(self.this_calc_dir,'LR_PandL')
            elif self.model_type == 'RFR':
                PandL_filename = os.path.join(self.this_calc_dir,'RFR_PandL')
            elif self.model_type == 'Lasso':
                PandL_filename = os.path.join(self.this_calc_dir,'Lasso_PandL')
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