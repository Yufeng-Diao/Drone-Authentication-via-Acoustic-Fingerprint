# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:22:25 2022

@author: Eidos
"""
import copy
import sys
import os
import wave
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)
    
import joblib
import numpy as np
import pandas as pd
import re
# from sklearn import discriminant_analysis
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn import neighbors
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm

import toolbox.info_detector_drone as idd
from toolbox.MFCC_extract import mfcc_extract
from toolbox.name_set import name_set_drone
from toolbox.name_set import name_set_csv
from toolbox.name_set import drone_set
from toolbox import audio_processing as ap
class Evaluator():
    
    def __init__(self, args):
        
        self.args = copy.deepcopy(args)
        # Intrinsic parameters
        self.audio = []
        self.audio_label = []
        self.audio_data = []
        # Carefully treat with this variable
        self.name_check = idd.FileNameProcessing(idd.name_set_drone)

        self.dic_quick_check(self.args.dic_choose, self.args.dic_aban, self.name_check)
        # Load all the valid audio
        self.audio, self.audio_label = self.audio_select(self.args.originData_path, 
                                          self.args.dic_choose, 
                                          self.args.dic_aban,
                                          self.name_check)
        if not self.audio:
            print('No drone available in this condition!')
            sys.exit()
        # Audio -> array
        for i in self.audio:
            self.audio_data.append(ap.audio_load(i))
        # Split the train set and evaluate set
        self.eval_data , self.eval_label = self.train_eval_split(self.audio_data, self.audio_label)
        
        # MFCC
        self.wave_feature_all, self.wave_label_all = \
            mfcc_extract(self.eval_data, self.eval_label, 
                         num_filter = self.args.mfcc['num_filter'],
                         num_cep = self.args.mfcc['num_cep'], 
                         winlen = self.args.mfcc['winlen'], 
                         winstep = self.args.mfcc['winstep'], 
                         fs = self.args.mfcc['fs'],
                         mfcc_d1_switch = self.args.mfcc['mfcc_d1_switch'], 
                         mfcc_d2_switch = self.args.mfcc['mfcc_d2_switch'])

    def model_load(self):
        print('Start loading model')
        classifier=joblib.load(self.args.model_path+'/'+self.args.model_name)
        print('Finish loading model')
        return classifier
        
    def save_mfcc_csv(self):
        # wave_feature_all_csv = pd.DataFrame(self.wave_feature_all)
        # wave_label_all_csv = pd.DataFrame(self.wave_label_all)
        
        # wave_feature_all_csv.to_csv(self.args.csv_savePath+'/'+self.args.csv_featureName, header=None, index=None)
        # wave_label_all_csv.to_csv(self.args.csv_savePath+'/'+self.args.csv_labelName, header=None, index=None)
        print('This function is abandoned!!!')
    
    def audio_select(self, originData_path, dic_choose, dic_aban, name_check):
        # Store audio data and label
        audio = []
        audio_label = []
        for root, dirs, files in os.walk(originData_path):
            for i in range(len(files)):
                name, label = name_check.info_detector(files[i],"drone_No")
                if name_check.check_file_choose(name, dic_choose) \
                    and not(name_check.check_file_aban(name, dic_aban)):
                        
                    audio.append(wave.open(root+'/'+files[i]))
                    audio_label.append(label)
                    print("Choose %s"%files[i])
                else:
                    print("abandon %s"%files[i])
                    pass
        return audio, audio_label

    def train_eval_split(self, audio_data, audio_label):
        # train_data = []
        eval_data = []
        
        # train_label = []
        eval_label = []
        for data in audio_data:
            seg_point = []
            seg_point.append(int(len(data)*0.2))
            seg_point.append(int(len(data)*0.35))
            seg_point.append(int(len(data)*0.65))
            seg_point.append(int(len(data)*0.8))
            
            # train_data.append(data[0:seg_point[0]])
            # train_data.append(data[seg_point[1]:seg_point[2]])
            # train_data.append(data[seg_point[3]:])
            
            eval_data.append(data[seg_point[0]:seg_point[1]])
            eval_data.append(data[seg_point[2]:seg_point[3]])
            
        for i in audio_label:
            # train_label.append(i)
            # train_label.append(i)
            # train_label.append(i)
            eval_label.append(i)
            eval_label.append(i)
        
        return eval_data, eval_label

    def dic_quick_check(self, dic_choose, dic_aban, name_check):
        # Check the format of the dic_choose
        if not(name_check.check_dic(dic_choose)):
            print("The format of the dic_choose is wrong!")
            sys.exit()
        # Check the format of the dic_aban
        if not(name_check.check_dic(dic_aban)):
            print("The format of the dic_aban is wrong!")
            sys.exit()
    
    

class Evaluator_csv(Evaluator):
    
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        
        # Carefully treat with this variable
        self.name_check = idd.FileNameProcessing(name_set_csv)
        # 
        self.wave_feature_all = None
        self.wave_label_all = None
        
        name_set_csv_list = ['prefix','date','num_filter','num_cep','winlen','winstep','multiset','suffix']
        dic_choose_csv = dict([(k,[]) for k in name_set_csv_list])
        dic_aban_csv = dict([(k,[]) for k in name_set_csv_list])
        # Do not change this. 
        dic_choose_csv['prefix'] = ['_label_']
        dic_choose_csv['date'] = self.args.dic_choose['date']
        
        # Check the format of the dic_choose_csv
        if not(self.name_check.check_dic(dic_choose_csv)):
            print("The format of the dic_choose_csv is wrong!")
            sys.exit()
        # Check the format of the dic_aban_csv
        if not(self.name_check.check_dic(dic_aban_csv)):
            print("The format of the dic_aban_csv is wrong!")
            sys.exit()
        
        # For detecting the range of drone set
        self.name_multiset = dict([(k,False) for k in name_set_csv['multiset']])
        
        # Load valid audio file
        for root, dirs, files in os.walk(self.args.csv_savePath):
            for i in range(len(files)):
                
                name, _ = self.name_check.info_detector(files[i], "suffix")
                if self.name_check.check_file_choose(name, dic_choose_csv) \
                    and not(self.name_check.check_file_aban(name, dic_aban_csv))\
                    and self.fileName_check(files[i]):
                    
                    # Drone set detection
                    for i_drone_set in name_set_csv['multiset']:
                        if self.name_check.check_file_choose(name, {'multiset':[i_drone_set]}):
                            self.name_multiset[i_drone_set] = True
                    
                    wave_label = np.array(pd.read_csv(root+'/'+files[i]))
                    wave_feature = np.array(pd.read_csv(root+'/'+files[i].replace('label','MFCC',1)))
                    # Here should design a drone select function in the future
                    wave_label, wave_feature = self.drone_selection(wave_label, wave_feature)
                    #**********************************************************#
                    #**********************************************************#
                    
                    try:
                        self.wave_feature_all = np.vstack((self.wave_feature_all, wave_feature))
                        self.wave_label_all = np.vstack((self.wave_label_all, wave_label))
                    except ValueError:
                        self.wave_feature_all = wave_feature
                        self.wave_label_all = wave_label
                        
                    print("Choose %s"%files[i])
                else:
                    print("abandon %s"%files[i])
                    pass
    
    def drone_selection(self, wave_label, wave_feature):
        label_dic = dict(zip(name_set_drone['drone_No'], 
                             np.arange(len(name_set_drone['drone_No']))))
        list_delete = []
        count = 0
        for label in wave_label:
            for name in self.args.dic_aban['drone_No']:
                if label == label_dic[name]:
                    list_delete.append(count)
                    break
            count = count + 1
        wave_label = np.delete(wave_label, list_delete, axis = 0)
        wave_feature = np.delete(wave_feature, list_delete, axis = 0)
        return wave_label, wave_feature
    
    def fileName_check(self, name):
        name_list = ['num_filter','num_cep','winlen','winstep']
        name_list_pattern = ['_.{1,4}nf_','_.{1,4}nc_','_.{1,4}wl_','_.{1,4}ws_']
        # Check the if MFCC parameters are valid
        for i in range(len(name_list_pattern)):
            if not (float(re.search('\d+\.?\d*',re.search(name_list_pattern[i], name).group()).group()) \
                    - float(self.args.mfcc[name_list[i]] < 0.000001)):
                print('This MFCC parameter is not right')
                print('fileName = ',float(re.search('\d+\.?\d*',re.search(name_list_pattern[i], name).group()).group()))
                print('train.py = ', float(self.args.mfcc[name_list[i]]))
                return False
        
        return True

class Evaluator_pkl(Evaluator_csv):
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        idx = pd.IndexSlice
        
        if not self.fileName_check(self.args.pkl_fileName):
            sys.exit()
        
        mfcc_list = ['0_mfcc']
        if self.args.mfcc['mfcc_d1_switch']:
            mfcc_list.append('1_mfcc')
        if self.args.mfcc['mfcc_d2_switch']:
            mfcc_list.append('2_mfcc')
        
        self.label_dic = dict(zip(name_set_drone['drone_No'], 
                                  np.arange(len(name_set_drone['drone_No']))))
        self.wave_label_all = []
        # Load the dataset
        pkl_dataset = pd.read_pickle(self.args.pkl_savePath+'/'+self.args.pkl_fileName)
        
        self.wave_feature_all = pkl_dataset.loc[idx['eval',self.args.dic_choose["date"],
                                                 :,self.args.dic_choose["drone_No"]],
                                                idx[mfcc_list,1:self.args.mfcc['num_cep']]]
        
        for multiIndex in self.wave_feature_all.index:
            self.wave_label_all.append(self.label_dic[multiIndex[-1]])
        
        self.wave_label_all = np.array(self.wave_label_all).reshape(-1,1)
        self.wave_feature_all = np.array(self.wave_feature_all)