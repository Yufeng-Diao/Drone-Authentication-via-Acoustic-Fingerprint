# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 22:10:48 2022

@author: Eidos
"""
from sklearn.metrics import accuracy_score

from runners.mid_runner_eval import Mid_runner_eval

class Mid_runner_eval_timeVar(Mid_runner_eval):
    def run(self):
        # Load the model
        self.model = self.runner.model_load()
        # Evaluate the model
        accuracy = self.test_evaluate()
        
        return accuracy
        # self.plot(label_pre)
    
    def test_evaluate(self):
        label_pre = self.model.predict(self.runner.wave_feature_all)
        label_true = self.runner.wave_label_all
        
        self.args.type_drone = self.drone_set_selection()
        self.args.label = self.label_generation(self.args.type_drone)
        print('drone_set_selection:', self.args.type_drone)
        
        accuracy = accuracy_score(label_true, label_pre)
       
        return accuracy