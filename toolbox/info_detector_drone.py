# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:57:40 2021

@author: MSI
"""

import numpy as np
import sys
import os
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)

from toolbox.name_set import name_set_drone

class FileNameProcessing():
    
    def __init__(self, name_set_dic):
        label_dic = {}
        for key in name_set_dic.keys():
            label_dic[key] = dict(zip(name_set_dic[key],
                                      np.arange(len(name_set_dic[key]))))
        self.name_set_dic = name_set_dic
        self.label_dic = label_dic
    
    def info_detector(self, fileName, name_key):
        '''
        
        Parameters
        ----------
        fileName : string
            The name of the target file, which should be saved in agreed format.
        name_key : string
            The property of the file we concerned. It decides name_target_label.
    
        Returns
        -------
        name_target : dic
            This dic saves the properties of the target file.
        name_target_label : int or NoneType
            It is the label of our concerned property.
    
        '''
        # add [] to make name_kay to list, 
        # so the set([name_key]) has only one element,
        # which is name_key.
        if not(set(self.name_set_dic.keys())>=set([name_key])):
            print("The format of the 'name_key' is incorrect!")
            sys.exit()
        fileName = fileName.lower()
        # for storing name string
        name_target = dict([(k,None) for k in self.name_set_dic.keys()])
        name_target_label = None
        # detect the valid part in the filename
        for key in name_target.keys():
            for name in self.name_set_dic[key]:
                if fileName.find(name) != -1:
                    name_target[key] = name
                    break
        # get the target label that we concerned
        try:        
            name_target_label = self.label_dic[name_key][name_target[name_key]]
        except KeyError:
            name_target_label = None
            
        return name_target, name_target_label
    
    def strict_abandon(self, name_target, name_target_label, 
                       switch_name=True, switch_label=True):
        """
        
        Parameters
        ----------
        name_target : dic
            DESCRIPTION.
        name_target_label : int
            DESCRIPTION.
        switch_name : bool, optional
            DESCRIPTION. The default is True.
        switch_label : bool, optional
            DESCRIPTION. The default is True.
    
        Returns
        -------
        bool
            True: abandon the file
            False: keep the file
    
        """
        switch_aban = False
        if switch_label:
            if name_target_label is None:
                switch_aban = True
        if switch_name:
            for key in name_target.keys():
                if name_target[key] is None:
                    switch_aban = True
        
        return switch_aban
    
    def check_file_choose(self, name_target, dic={}):
        """
        
    
        Parameters
        ----------
        name_target : dic
            The dic of the name.
        dic : dic, optional
            The dic of the selected propeties. The default is {}.
    
        Returns
        -------
        bool
            True: this file has selected properties.
            False: this file does not have selected properties.
    
        """
        for key in dic.keys():
            if dic[key] == []:
                continue
            # choose or abandon the file according to target properties
            if not(set(dic[key])>=set([name_target[key]])):
                return False
        
        return True
    
    def check_file_aban(self, name_target, dic={}):
        """
        
    
        Parameters
        ----------
        name_target : dic
            The dic of the name.
        dic : dic, optional
            The dic of the abandoned propeties. The default is {}.
    
        Returns
        -------
        bool
            True: this file has abandoned properties.
            False: this file does not have abandoned properties.
    
        """
        for key in dic.keys():
            if dic[key] == []:
                continue
            # choose or abandon the file according to target properties
            if set(dic[key])>=set([name_target[key]]):
                return True
        
        return False
          
    def check_dic(self, dic_use):
        """
        
    
        Parameters
        ----------
        dic_use : dic
            DESCRIPTION.
    
        Returns
        -------
        bool
            DESCRIPTION.
    
        """
        indicator_key = False
        indicator_list = False
        # determine if dic_use has valid keys
        if set(self.name_set_dic.keys()) >= set(dic_use.keys()):
            indicator_key = True
        # determine if dic_use has valid lists
        try:
            for key in dic_use.keys():
                if set(self.name_set_dic[key]) >= set(dic_use[key]):
                    indicator_list = True
                else:
                    print("The list inside the dic is wrong")
                    indicator_list = False
                    break
        except KeyError:
            indicator_list = False
        
        if indicator_key and indicator_list:
            return True
        else:
            return False


if __name__ == "__main__":
    
    string_1 = '_UAV_20210804_d4_hover_1m_1_.WAV'
    string_2 = '20210804_d4_hover_1m_1_.WAV'
    string_3 = '_UAV_20210803_d3_ud_5m_1_.WAV'
    # name_target_1, name_target_label_2 = info_detector(string_1,'drone_No')
    # name_target_2, name_target_label_2 = info_detector(string_2,'date')
    name_check = FileNameProcessing(name_set_drone)
    name_target_1, name_target_label_1 = name_check.info_detector(string_1,'drone_No')
    name_target_2, name_target_label_2 = name_check.info_detector(string_2,'prefix')
    print(name_check.strict_abandon(name_target_1,name_target_label_1))
    print(name_check.strict_abandon(name_target_2,name_target_label_2))
    
    dic = {}
    dic["prefix"]=["_uav_"]
    dic["state"]=["_hover_"]
    
    print(name_check.check_file_choose(name_target_1,dic))
    print(name_check.check_file_choose(name_target_2,dic))
    