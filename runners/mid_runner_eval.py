# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:07:36 2022

@author: Eidos
"""
import numpy as np
import os
import sys
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)
    
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from runners.runner_eval import Evaluator
from runners.runner_eval import Evaluator_csv
from runners.runner_eval import Evaluator_pkl
from toolbox import CMatrix
from toolbox.name_set import name_set_drone

sys.path.append(r"E:\1_Research\3_UAV_2\4_pyeidos_drone\3_GitKraken_development\drone_authentication\runners")

class Mid_runner_eval():
    
    def __init__(self, args):
        self.args = args
        # Read saved csv files of raw data
        if self.args.csv_use and not self.args.pkl_use:
            self.runner = Evaluator_csv(self.args)
        elif not self.args.csv_use and self.args.pkl_use:
            self.runner = Evaluator_pkl(self.args)
        else:
            self.runner = Evaluator(self.args)
        # Save mfcc features and labels in csv files
    
    def run(self):
        # Load the model
        self.model = self.runner.model_load()
        # Evaluate the model
        label_pre = self.test_evaluate()
        
        self.plot(label_pre)
        
    def test_evaluate(self):
        label_pre = self.model.predict(self.runner.wave_feature_all)
        label_true = self.runner.wave_label_all
        
        self.args.type_drone = self.drone_set_selection()
        self.args.label = self.label_generation(self.args.type_drone)
        print('drone_set_selection:', self.args.type_drone)
        try:
            print(classification_report(label_true, label_pre, 
                                        labels = self.args.label, 
                                        target_names = self.args.type_drone,
                                        zero_division = 0 ,
                                        digits = 4))
        except:
            print('Number of classes does not match size of target_names!')
        print(np.around(confusion_matrix(label_true.reshape(-1,1), label_pre.reshape(-1,1), normalize = 'true'),3))
        return label_pre
    
    def plot(self, label_pre):
        # label = np.array(self.args.label)+1
        CMatrix(self.runner.wave_label_all, label_pre, self.args.label, self.args.type_drone)
    
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