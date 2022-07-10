# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:44:05 2021

@author: Eidos
"""

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def CMatrix(y_true,y_pred, typeNameSet, axis_label):
    
    sns.set()
    plt.rc('font',family='Times New Roman')
    # f,ax=plt.subplots(figsize=(6.4*3,4.8*3))
    f,ax=plt.subplots(figsize=(8,6))
    C2 = confusion_matrix(y_true, y_pred, labels=typeNameSet, normalize = 'true')
    C2 = np.around(C2, 3)
    #C2= confusion_matrix(y_true, y_pred, labels=[6,8,9], normalize = 'true')
    sns.heatmap(C2,annot=True,ax=ax) 
    ax.set_xticklabels(axis_label)
    ax.set_yticklabels(axis_label)
    # ax.set_title('confusion matrix',fontsize = 20) 
    ax.set_xlabel('Predict Drone No.',fontsize = 15) 
    ax.set_ylabel('True Drone No.',fontsize = 15) 
    plt.tight_layout()
    
if __name__ == "__main__":
    
    y_true = np.array([0,0,1,1,2,2,0,1])
    y_pred = np.array([1,0,0,1,1,2,1,0])
    typeNameSet = ['U1','U2','U3','U4']
    CMatrix(y_true,y_pred,typeNameSet)