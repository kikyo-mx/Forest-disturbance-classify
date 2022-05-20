#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# data1 = pd.read_csv('train.csv',header=0)
def plot_feature_importance(clf, feature_names):
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[-10:]
    pos = np.arange(sorted_idx.shape[0]) + 4.5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx],height=0.3,align='center')
    # plt.yticks(pos, feature_names[sorted_idx])
    plt.yticks(pos, [feature_names[idx] for idx in sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

    
def get_result(conf):
    print(conf)
    recall = conf[0][0] / conf[0].sum() * 100
    precision = conf[0][0] / conf[:, 0].sum() * 100
    accuracy = (conf[0][0] + conf[1][1]) / conf.sum() * 100
    F1 = 2 * recall * precision / (recall + precision)
    print('recall : %.4f %%' % recall)
    print('precision : %.4f %%' % precision)
    print('accuracy : %.4f %%' % accuracy)
    print('F1 : %.4f' % F1)
    print([recall, precision, accuracy, F1])
    return [recall, precision, accuracy, F1]


class Tree:
    def __init__(self, feature_list, train_csv, test_csv):
        self.feature_list = feature_list
        self.data_train = pd.read_csv(train_csv)
        self.data_test = pd.read_csv(test_csv)
    def decision_result(self):
        print('decision_tree')
        return DecisionTreeClassifier(max_depth =10,min_samples_split=40,min_samples_leaf=10)
    def random_result(self):
        print('random_forest')
        return RandomForestClassifier(n_estimators=300,max_depth =10,min_samples_split=40,min_samples_leaf=10)
    def adaboost_result(self):
        print('adaboost')
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=40, min_samples_leaf=10),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
    def process(self, model):
        result_ABC = []
        for feature in self.feature_list:
            train_data = self.data_train[feature].values
            train_label = self.data_train['label'].values - 1
            test_data = self.data_test[feature].values
            test_label = self.data_test['label'].values - 1
            rfc = model.fit(train_data,train_label)
            joblib.dump(rfc, "test_model.model")
            prediction = rfc.predict(test_data)
            conf = confusion_matrix(test_label, prediction)
            result = get_result(conf)
            result_ABC.append(result)
            plot_feature_importance(clf=rfc, feature_names=feature)
        return result_ABC
        
# joblib.dump(rfc,'AdaBoost_data_major_two_random.model')

