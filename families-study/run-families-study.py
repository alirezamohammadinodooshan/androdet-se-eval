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
from tqdm import tqdm


PROJECT_ROOT_DIR = path.abspath('..')
sys.path.append(path.join(PROJECT_ROOT_DIR, 'tools'))
from csv_to_arff import csv2arff

def generate_data_set():
    rand_stat = 42  # for replication purposes
    families = [family.rstrip('\n') for family in open('AMD-families-sorted-by-size.txt')]
    amd_androdet_data = pd.read_csv(path.join(PROJECT_ROOT_DIR, "amd-androdet/AMD-androdet-features.csv"))
    if not path.exists('./dataset/'):  # create the folder for the families data if not exists
        makedirs('./dataset/')
    current_dir = path.abspath('.')
    for family in tqdm(families):
        test_data = amd_androdet_data[amd_androdet_data['Family'] == family]
        train_data = amd_androdet_data[amd_androdet_data['Family'] != family]
        family_data_dir = '{}/dataset/{}'.format(current_dir, family)
        if not path.exists(family_data_dir):
            makedirs(family_data_dir)
        train_data_file = path.join(family_data_dir, 'train.csv')
        test_data_file = path.join(family_data_dir, 'test.csv')
        train_data_file_4_atm = path.join(family_data_dir, 'atm_train.csv')
        test_data_file_4_atm = path.join(family_data_dir, 'atm_test.csv')
        train_data = train_data.sample(frac=1, random_state=rand_stat)  # shuffling the training data
        test_data = test_data.sample(frac=1, random_state=rand_stat)
        train_data.to_csv(train_data_file, index=False)
        test_data.to_csv(test_data_file, index=False)
        train_data.iloc[:,3:].to_csv(train_data_file_4_atm, index=False)
        test_data.iloc[:,3:].to_csv(test_data_file_4_atm, index=False)
        csv2arff(train_data_file_4_atm, family)
        csv2arff(test_data_file_4_atm, family)


def do_moa_eval():
    families = [family.rstrip('\n') for family in open('AMD-families-sorted-by-size.txt')]
    results_list = []
    results_list.append(('family', 'tp', 'tn', 'fp', 'fn'))
    current_dir = path.abspath('.')
    for family in tqdm(families):
        family_data_dir = '{}/dataset/{}'.format(current_dir, family)
        train_file = path.join(family_data_dir, 'train.arff')
        test_file = path.join(family_data_dir, 'test.arff')
        return_str = subprocess.check_output("java -cp {}/tools/moa:{}/tools/moa/moa.jar \
            moaeval.eval {} {}".format(PROJECT_ROOT_DIR, PROJECT_ROOT_DIR, train_file, test_file), shell=True)
        return_str = return_str.decode("utf-8").strip()
        tp = return_str.split(',')[0]
        tn = return_str.split(',')[1]
        fp = return_str.split(',')[2]
        fn = return_str.split(',')[3]
        results_list.append((family, tp, tn, fp, fn))
    with open("moa_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_list)


def train_atm_model(family): 
    current_dir = path.abspath('.')
    family_data_dir = '{}/dataset/{}'.format(current_dir, family)
    train_file = path.join(family_data_dir, 'atm_train.csv')
    np.random.seed(0)  # for replication purposes
    random.seed(0)
    atm_models_dir = '{}/atm-models'.format(current_dir)
    family_atm_model_dir = '{}/{}'.format(atm_models_dir, family)
    if not path.exists(family_atm_model_dir): 
        makedirs(family_atm_model_dir)
    chdir(family_atm_model_dir)
    clf = ATM()
    results = clf.run(train_path=train_file, name=family,
        budget=200, budget_type='classifier', metric='accuracy', methods=['svm'])
    best_classifier_pkl_file = '{}/model_{}.pkl'.format(family_atm_model_dir, family)
    results.export_best_classifier(best_classifier_pkl_file)
    chdir(current_dir)
    return family


def do_atm_eval():
    families = [family.rstrip('\n') for family in open('AMD-families-sorted-by-size.txt')]
    results_list = []
    results_list.append(('family', 'tp', 'tn', 'fp', 'fn'))
    current_dir = path.abspath('.')
    not_yet_trained_families = []
    atm_models_dir = '{}/atm-models'.format(current_dir)
    if not path.exists(atm_models_dir): 
        makedirs(atm_models_dir)
    for family in families:
        model_pkl_file = '{}/{}/model_{}.pkl'.format(atm_models_dir, family, family)
        if not path.exists(model_pkl_file):
            not_yet_trained_families.append(family)
    if not_yet_trained_families:
        pool = multiprocessing.Pool(processes=71)  # start 71 worker processes
        result = [pool.apply_async(train_atm_model, (family,)) for family in not_yet_trained_families]
        for elem in result:
            print("Training a model for the {} family is done!".format(elem.get()))

    for family in tqdm(families):
        y_true = []
        y_pred = []
        family_data_dir = '{}/dataset/{}'.format(current_dir, family)
        test_file = path.join(family_data_dir, 'atm_test.csv')
        model_pkl_file = '{}/atm-models/{}/model_{}.pkl'.format(current_dir, family, family)
        model = Model.load(model_pkl_file)
        testing_data = pd.read_csv(test_file)
        pred = model.predict(testing_data)
        y_true = list(testing_data['class'])
        y_pred += list(pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        results_list.append((family, tp, tn, fp, fn))
    with open("atm_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="python code to do the families study as described in the comment paper")
    parser.add_argument('--dataset', help="To generate the data set for each of the families\
        (if not passed will use the previously generated dataset in the dataset folder for the moa and atm analysis)", action='store_true')
    parser.add_argument('--do_moa', help="To do the moa analysis ", action='store_true')
    parser.add_argument('--do_atm', help="To do the atm analysis ", action='store_true')
    args = parser.parse_args()
    if args.dataset is True:
        generate_data_set()
    if args.do_moa is True:
        do_moa_eval()
    if args.do_atm is True:
        do_atm_eval()