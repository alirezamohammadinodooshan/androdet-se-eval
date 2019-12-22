#!/usr/bin/env python3.7

import argparse
import pandas as pd
import sys
from os import path as path
from os import makedirs, chdir
import subprocess
import csv
from atm import ATM
from atm import Model
from sklearn import metrics
import numpy as np
import random
import multiprocessing
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT_DIR = path.abspath('..')
sys.path.append(path.join(PROJECT_ROOT_DIR,'tools'))
from csv_to_arff import csv2arff


def generate_data_set(test_ratio):
    amd_androdet_data = pd.read_csv(path.join(PROJECT_ROOT_DIR, "amd-androdet/AMD-androdet-features.csv"))
    if not path.exists('./dataset/'): # create the folder for each cross validation data if not exists
        makedirs('./dataset/')
    current_dir = path.abspath('.')
    cross_validations = range(100)
    for i in tqdm(cross_validations):  # generating the data for 100 cross validations
        np.random.seed(i)
        train_data = amd_androdet_data.sample(frac=test_ratio, random_state=i) 
        test_data = amd_androdet_data.drop(train_data.index)
        cross_validation_i_data_dir = '{}/dataset/{}'.format(current_dir, i)
        if not path.exists(cross_validation_i_data_dir): 
            makedirs(cross_validation_i_data_dir)
        train_data_file = path.join(cross_validation_i_data_dir, 'train.csv')
        test_data_file = path.join(cross_validation_i_data_dir, 'test.csv')
        train_data_file_4_atm = path.join(cross_validation_i_data_dir, 'atm_train.csv')
        test_data_file_4_atm = path.join(cross_validation_i_data_dir, 'atm_test.csv')
        train_data.to_csv(train_data_file, index=False)
        test_data.to_csv(test_data_file, index=False)
        train_data.iloc[:,3:].to_csv(train_data_file_4_atm, index=False)
        test_data.iloc[:,3:].to_csv(test_data_file_4_atm, index=False)
        csv2arff(train_data_file_4_atm, str(i))
        csv2arff(test_data_file_4_atm, str(i))


def do_moa_eval():
    cross_validations = range(100)
    results_list = []
    results_list.append(('cross_validation', 'tp', 'tn', 'fp', 'fn'))
    for cross_validation in tqdm(cross_validations):
        cross_validation_data_dir = '{}/random-study/dataset/{}'.format(PROJECT_ROOT_DIR, cross_validation)
        train_file = path.join(cross_validation_data_dir, 'train.arff')
        test_file = path.join(cross_validation_data_dir, 'test.arff')
        return_str = subprocess.check_output("java -cp {}/tools/moa:{}/tools/moa/moa.jar \
            moaeval.eval {} {}".format(PROJECT_ROOT_DIR, PROJECT_ROOT_DIR, train_file, test_file), shell=True)
        return_str = return_str.decode("utf-8").strip()
        tp = return_str.split(',')[0]
        tn = return_str.split(',')[1]
        fp = return_str.split(',')[2]
        fn = return_str.split(',')[3]
        results_list.append((cross_validation, tp, tn, fp, fn))
    with open("moa_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_list)

def train_atm_model(cross_validation): 
    current_dir = path.join(PROJECT_ROOT_DIR, "random-study")
    cross_validation_data_dir = '{}/dataset/{}'.format(current_dir, cross_validation)
    train_file = path.join(cross_validation_data_dir, 'atm_train.csv')
    np.random.seed(0)  # for replication purposes
    random.seed(0)
    atm_models_dir = '{}/atm-models'.format(current_dir)
    cross_validation_atm_model_dir = '{}/{}'.format(atm_models_dir, cross_validation)
    if not path.exists(cross_validation_atm_model_dir): 
        makedirs(cross_validation_atm_model_dir)
    chdir(cross_validation_atm_model_dir)
    clf = ATM()
    results = clf.run(train_path=train_file, name=cross_validation,
        budget=200, budget_type='classifier', metric='accuracy', methods=['svm'])
    best_classifier_pkl_file = '{}/model_{}.pkl'.format(cross_validation_atm_model_dir, cross_validation)
    results.export_best_classifier(best_classifier_pkl_file)
    chdir(current_dir)
    return cross_validation


def do_atm_eval():
    cross_validations = range(100)
    results_list = []
    results_list.append(('cross_validation', 'tp', 'tn', 'fp', 'fn'))
    current_dir = path.join(PROJECT_ROOT_DIR, "random-study")
    not_yet_trained_cross_validations = []
    atm_models_dir = '{}/atm-models'.format(current_dir)
    if not path.exists(atm_models_dir): 
        makedirs(atm_models_dir)
    for cross_validation in cross_validations:
        model_pkl_file = '{}/{}/model_{}.pkl'.format(atm_models_dir, cross_validation, cross_validation)
        if not path.exists(model_pkl_file):
            not_yet_trained_cross_validations.append(cross_validation)
    if not_yet_trained_cross_validations:
        pool = multiprocessing.Pool(processes=100)  # start 100 worker processes
        result = [pool.apply_async(train_atm_model, (cross_validation,)) for cross_validation in not_yet_trained_cross_validations]
        for elem in result:
            print("Training a model for the {} cross_validation is finished!".format(elem.get()))

    for cross_validation in tqdm(cross_validations):
        y_true = []
        y_pred = []
        cross_validation_data_dir = '{}/dataset/{}'.format(current_dir, cross_validation)
        test_file = path.join(cross_validation_data_dir, 'atm_test.csv')
        model_pkl_file = '{}/atm-models/{}/model_{}.pkl'.format(current_dir, cross_validation, cross_validation)
        model = Model.load(model_pkl_file)
        testing_data = pd.read_csv(test_file)
        pred = model.predict(testing_data)
        y_true = list(testing_data['class'])
        y_pred += list(pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        results_list.append((cross_validation, tp, tn, fp, fn))
    with open("atm_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="python code to do the random study as described in the comment paper")
    parser.add_argument('--dataset', help="To generate the data set for each of the cross validations\
        (if false will use the current generated dataset in the dataset folder for the moa and atm analysis)",action='store_true')
    parser.add_argument('--do_moa', help="To do the moa analysis ", action='store_true')
    parser.add_argument('--do_atm', help="To do the atm analysis ", action='store_true')
    args = parser.parse_args()
    if args.dataset is True:
        generate_data_set(0.5)
    if args.do_moa is True:
        do_moa_eval()
    if args.do_atm is True:
        do_atm_eval()
