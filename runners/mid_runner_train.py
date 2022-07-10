# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:07:36 2022

@author: Eidos
"""
import os
import sys
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)
    
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from runners.runner_train import Trainer
from runners.runner_train import Trainer_csv
from runners.runner_train import Trainer_pkl
from toolbox.name_set import name_set_drone

class Mid_runner_train():
    def __init__(self, args):
        self.args = args
        # Read saved csv files of raw data
        if self.args.csv_use and not self.args.pkl_use:
            self.runner = Trainer_csv(self.args)
        elif not self.args.csv_use and self.args.pkl_use:
            self.runner = Trainer_pkl(self.args)
        else:
            self.runner = Trainer(self.args)
    
    def run(self):
        # Train the model
        self.model_train = self.classifier_use()
        # Evaluate the model
        # self.train_evaluate()
        # Save the model
        if self.args.model_save:
            self.runner.save_model(self.model_train)
        
    def classifier_use(self):
        if self.args.qda:
            model_train = self.runner.qda_train()
        elif self.args.lda:
            model_train = self.runner.lda_train()
        elif self.args.lsvm:
            model_train = self.runner.lsvm_train()
        elif self.args.svm:
            model_train = self.runner.svm_train()
        elif self.args.knn:
            model_train = self.runner.knn_train()
        elif self.args.dt:
            model_train = self.runner.dt_train()
        elif self.args.rf:
            model_train = self.runner.rf_train()
        elif self.args.gnb:
            model_train = self.runner.gnb_train()
        else:
            model_train = None
        return model_train
    
    def train_evaluate(self):
        try:
            print('Classification report')
            label_pre = self.model_train.predict(self.runner.wave_feature_all)
            label_true = self.runner.wave_label_all
            self.args.type_drone = self.drone_set_selection()
            self.args.label = self.label_generation(self.args.type_drone)
            print('drone_set_selection:', self.args.type_drone)
            print('label_generation', self.args.label)
            print(classification_report(label_true, label_pre, 
                                        labels = self.args.label, 
                                        target_names = self.args.type_drone))
        except ValueError:
            print('Number of classes does not match size of target_names!')
        print('\nconfusion matrix')
        print(np.around(confusion_matrix(label_true, label_pre, normalize = 'true'),3))
    
    def drone_set_selection(self):
        label_list = name_set_drone['drone_No'].copy()

        for name in label_list:
            if name in self.args.dic_aban['drone_No']:
                label_list.remove(name)
        return label_list
    
    def label_generation(self, label_list):
        label_dic = dict(zip(name_set_drone['drone_No'], 
                             np.arange(len(name_set_drone['drone_No']))))
        label = []
        for name in label_list:
            label.append(label_dic[name])
        
        return label