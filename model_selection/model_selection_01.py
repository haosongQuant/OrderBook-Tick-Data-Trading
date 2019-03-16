#coding=utf-8
__author__ = 'haosong'

import os, sys
import pandas as pd
import numpy as np
import time
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,\
                              GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..//utils')
from parsePara import parsePara

def read_csv(path, product, day_trade):
    data_label_feature = []
    for k in np.arange(0,len(day_trade)):
        read_path = os.path.join(path, product + '_' + day_trade[k]  + '_label_feature.csv')
        data_label_feature.append(pd.read_csv(read_path))
    return data_label_feature

def ml_algos():
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state = 0),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state = 0),
        'AdaBoostClassifier': AdaBoostClassifier(base_estimator = DecisionTreeClassifier(),\
                                                 n_estimators = 10,random_state = 0),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state = 0),
        #'SVC': SVC(probability=True,random_state = 0),
    }
    model_grid_params = {
        'RandomForestClassifier': {'max_features':[None],'n_estimators':[10],'max_depth':[10],\
                                   'min_samples_split':[2],'criterion':['entropy'],\
                                   'min_samples_leaf':[3]},
        'ExtraTreesClassifier': {'max_features':[None],'n_estimators':[10],'max_depth':[10],\
                                 'min_samples_split':[2],'criterion':['entropy'],\
                                 'min_samples_leaf':[3]},
        'AdaBoostClassifier': {"base_estimator__criterion" : ["entropy"],\
                               "base_estimator__max_depth": [None],\
                               "base_estimator__min_samples_leaf" : [3],\
                               "base_estimator__min_samples_split" : [2],\
                               "base_estimator__max_features" : [None]},
        'GradientBoostingClassifier': {'max_features':[None],'n_estimators':[10],'max_depth':[10],\
                                       'min_samples_split':[2],'min_samples_leaf':[3],\
                                       'learning_rate':[0.1],'subsample':[1.0]},
        'SVC': [{'kernel':['rbf'],'gamma':[1e-1],'C':[1]},\
                {'kernel':['linear'],'C':[1,10]}]
    }
    return models, model_grid_params

class Model_Selection:
    
    def __init__(self,models,model_grid_params,data_set,latest_sec,pred_sec,day, traded_time):

        self.models = models
        self.model_grid = model_grid_params
        self.data_set = data_set
        self.latest_sec = latest_sec # 训练时长
        self.pred_sec = pred_sec     # 预测时长
        self.day = day               # 测试天数
        self.traded_time = traded_time #最长持仓时间
        self.keys = models.keys()    # 分类器名称
        self.best_score = {}
        self.grid = {}
        self.predict_values = {}
        self.cv_acc = {}
        self.acc = {}
        self.fscore = {}
        self.true_values = {}
        self.predict_values_day = {}
        self.cv_acc_day = {}
        self.acc_day = {}
        self.fscore_day = {}
        self.true_values_day = {}
        self.summary_day = []

        self.quoteSpread = []
        self.short_loss = []
        self.long_loss = []

    def Grid_fit(self,X_train,y_train,cv = 5,scoring = 'accuracy'):

        for key in self.keys:
            print("Running GridSearchCV for %s." %(key) )
            model = self.models[key]
            model_grid = self.model_grid[key]
            Grid = GridSearchCV(model, model_grid, cv = cv, scoring = scoring)
            Grid.fit(X_train,y_train)
            self.grid[key] = Grid
            print(Grid.best_params_)
            print('CV Best Score = %s'%(Grid.best_score_))
            self.cv_acc[key].append(Grid.best_score_)

    def model_fit(self,X_train, y_train, X_test, y_test):

        for key in self.keys:
            print("Running training & testing for %s." %(key))
            model = self.models[key]
            model.set_params(**self.grid[key].best_params_)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            self.predict_values[key].append(predictions.tolist())
            self.true_values[key].append(y_test.tolist())
            acc = metrics.accuracy_score(y_test,predictions)
            f_score = metrics.f1_score(y_test,predictions)
            print('Accuracy = %s'%(acc))
            self.acc[key].append(acc)
            self.fscore[key].append(f_score)

            if key == 'SVC':
                if self.grid[key].best_params_.values()[0] == 'linear':
                    feature_imp = dict(zip([i for i in range(0,64,1)],model.coef_[0]))
                    Top_five = sorted(feature_imp.items(),key = lambda x : x[1] , reverse=True)[0:5]
                    #print('Kernel is linear and top five importance features = %s'%(Top_five))
                else:
                    #print('Kernel is rbf'
                    pass
            else:
                feature_imp = dict(zip([i for i in range(0,64,1)],model.feature_importances_))
                Top_five = sorted(feature_imp.items(),key = lambda x : x[1] , reverse=True)[0:5]
                #print('Top five importance features = %s'%(Top_five))
                pass

    def pipline(self):

        self.set_list_day() # store day values
        for day in np.arange(0,self.day):
            self.set_list() # store values
            print('Day = %s'%(day+1))
            quoteSpreadBuf = []
            short_lossBuf = []
            long_lossBuf = []
            for i in np.arange(0,len(self.data_set[day])-self.latest_sec-self.traded_time,self.pred_sec):

                print('--------------------Rolling Window Time = %s--------------------'%(i/self.pred_sec))
                # Train data
                data_train = self.data_set[day][i:i+self.latest_sec]
                X_train = data_train.drop(['0','65','66','67'],axis=1)
                y_train = data_train['0']

                # Test data
                data_test = self.data_set[day][i + self.latest_sec:i + self.latest_sec + self.pred_sec]
                X_test = data_test.drop(['0','65','66','67'],axis=1)
                y_test = data_test['0']

                quoteSpreadBuf.extend(list(data_test['65']))
                for k in np.arange(i + self.latest_sec, i + self.latest_sec + self.pred_sec):
                    loss1 = self.data_set[day]['67'][k] - self.data_set[day]['66'][k+self.traded_time] if (k+self.traded_time) < len(self.data_set[day]) \
                            else self.data_set[day]['67'][k] - self.data_set[day]['66'][-1]
                    short_lossBuf.append(loss1)
                    loss2 = self.data_set[day]['67'][k+self.traded_time] - self.data_set[day]['66'][k] if (k+self.traded_time) < len(self.data_set[day]) \
                            else self.data_set[day]['67'][-1] - self.data_set[day]['66'][k]
                    long_lossBuf.append(loss2)

                #start = time.time()
                self.Grid_fit(X_train, y_train, cv = 5, scoring = 'accuracy')
                self.model_fit(X_train, y_train,X_test,y_test)
                #end = time.time()
                #print('Total Time = %s'%(end - start)

            self.quoteSpread.append(quoteSpreadBuf)
            self.short_loss.append(short_lossBuf)
            self.long_loss.append(long_lossBuf)

            for key in self.keys:

                self.cv_acc_day[key].append(self.cv_acc[key])
                self.acc_day[key].append(self.acc[key])
                self.fscore_day[key].append(self.fscore[key])
                self.true_values_day[key].append(self.true_values[key])
                self.predict_values_day[key].append(self.predict_values[key])

            # self.summary_day.append(self.score_summary(sort_by = 'Accuracy_mean'))

    def set_list(self):

        for key in self.keys:
            self.predict_values[key] = []
            self.cv_acc[key] = []
            self.acc[key] = []
            self.fscore[key] = []
            self.true_values[key] = []

    def set_list_day(self):

        for key in self.keys:
            self.predict_values_day[key] = []
            self.cv_acc_day[key] = []
            self.acc_day[key] = []
            self.fscore_day[key] = []
            self.true_values_day[key] = []

    def score_summary(self,sort_by):

        summary = pd.concat([pd.DataFrame(self.acc.keys()),pd.DataFrame(map(lambda x: np.mean(self.acc[x]), self.acc)),\
                             pd.DataFrame(map(lambda x: np.std(self.acc[x]), self.acc)),\
                             pd.DataFrame(map(lambda x: max(self.acc[x]), self.acc)),\
                             pd.DataFrame(map(lambda x: min(self.acc[x]), self.acc)),\
                             pd.DataFrame(map(lambda x: np.mean(self.fscore[x]), self.fscore))],axis=1)
        summary.columns = ['Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score']
        summary.index.rename('Ranking', inplace=True)
        return summary.sort_values(by = [sort_by], ascending=False)

    def print_(self):
        print(self.predict_values)


if __name__ == '__main__':

    parms = parsePara()

    # datasetPath = 'C:\\Users\\haosong\\Documents\\OrderBook-Tick-Data-Trading\\train_test_builder_for_DDQuote'
    datasetPath = 'D:\\OrderBook-Tick-Data-Trading\\train_test_builder_for_DDQuote'
    product = parms['contract']
    datelist = parms['datelist']
    day_trade = datelist.split(',')

    # data_label_feature = read_csv(datasetPath, product, day_trade)
    models, model_grid_params = ml_algos()

    latest_sec = 60 * 30
    pred_sec = 10
    traded_time = 60 * 10
    # day = len(day_trade)
    day = 1

    for iDay in np.arange(len(day_trade)):

        print('---------- ', product, ' - ', day_trade[iDay], ' ----------')
        data_label_feature = read_csv(datasetPath, product, [day_trade[iDay]])

        pip = Model_Selection(models,model_grid_params,data_label_feature,latest_sec,pred_sec,day, traded_time)
        start = time.time()
        pip.pipline()
        end = time.time()
        print('Total Time = %s'%(end-start))

        pipKeyList = list(pip.keys)
        # compute cum_profit and Best_cv_score
        dict_ = {}
        dict_['cum_profit'] = []
        dict_['Best_cv_score'] = []

        cum_profit_label = []
        cum_profit = []
        best_cv_score = []

        spread = pip.quoteSpread[0][9::10]
        loss = pip.short_loss[0][9::10]

        for j in np.arange(0,len((pip.cv_acc_day[pipKeyList[0]])[0])):
            max_al = {}
            for i in np.arange(0,len(pipKeyList)):
                max_al[pipKeyList[i]] = np.array(pip.cv_acc_day[pipKeyList[i]])[0][j]
            # select best algorithm in cv = 5
            top_cv_acc = sorted(max_al.items(),key = lambda x : x[1], reverse = True)[0:1][0]
            best_cv_score.append(top_cv_acc[1])
            submission = pip.predict_values_day[top_cv_acc[0]][0][j][-1]
            true_value = pip.true_values_day[top_cv_acc[0]][0][j][-1]

            if submission == true_value:
                if submission == 1:
                    cum_profit_label.append(1)
                    cum_profit.append(spread[j])
                elif submission == 0:
                    cum_profit_label.append(0)
                    cum_profit.append(0)
            elif submission != true_value:
                if submission == 1:
                    cum_profit_label.append(-1)
                    cum_profit.append(loss[j])
                elif submission == 0:
                    cum_profit_label.append(0)
                    cum_profit.append(0)

        dict_['cum_profit'].append(cum_profit)
        dict_['Best_cv_score'].append(best_cv_score)

        sns.set_style("whitegrid")
        plt.figure(figsize = (20,8))
        plt.subplot(211)
        plt.plot(cum_profit,'-o',label = 'Profit & Loss',lw = 1,markersize = 3)
        plt.ylabel('Tick',size = 15)
        plt.legend(loc=0)
        plt.ylim(-7.5,2.5)
        plt.subplot(212)
        plt.plot(np.cumsum(cum_profit),'-o',label = 'Cum Profit',lw = 1,markersize = 2)
        plt.legend(loc=0)
        plt.xlabel('Rolling Window Numbers',size = 15)
        plt.ylabel('Profit',size = 15)
        # plt.show()

        savePath = os.path.join('D:\\OrderBook-Tick-Data-Trading\\model_selection',
                                product+'_'+day_trade[iDay]+'.png')
        plt.savefig(savePath)
        plt.close()