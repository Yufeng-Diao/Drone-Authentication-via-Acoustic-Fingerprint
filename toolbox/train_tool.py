# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 17:56:40 2022

@author: Eidos
"""


def model_name(args):
    if args.qda:
        model_name = '_QDA_'
    elif args.lda:
        model_name = '_LDA_'
    elif args.lsvm:
        model_name = '_LSVM_'
    elif args.svm:
        model_name = '_SVM_'
    elif args.knn:
        model_name = '_KNN_'
    elif args.dt:
        model_name = '_DT_'
    elif args.rf:
        model_name = '_RF_'
    elif args.gnb:
        model_name = '_GNB_'
    else:
        model_name = None
    # Change a little for using different number of features
    name_output = '%s%inf_%inc_%.2fwl_%.2fws_attack'%(model_name,
                                                args.mfcc['num_filter'],
                                                args.mfcc['num_cep'],
                                                args.mfcc['winlen'],
                                                args.mfcc['winstep'])

    name_output = name_output + '.m'
    return name_output

def mutual_exclusive(args, pos):
    me_list = [False, False, False, False, False, False, False, False]
    me_list[pos] = True
    
    args.qda = me_list[0]
    args.lda = me_list[1]
    args.lsvm = me_list[2]
    args.svm = me_list[3]
    args.knn = me_list[4]
    args.dt = me_list[5]
    args.rf = me_list[6]
    args.gnb = me_list[7]