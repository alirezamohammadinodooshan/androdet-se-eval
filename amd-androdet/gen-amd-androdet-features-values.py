#!/usr/bin/env python2.7

import pandas as pd
import sys
from os import path, makedirs
import argparse

PROJECT_ROOT_DIR = path.abspath('..')
sys.path.append(path.join(PROJECT_ROOT_DIR, 'tools'))
import file_tools, EFS


def generate_amd_androdet_dataset(amd_apks_folder, dexdump_dir):
    amd_androdet_dataset = pd.DataFrame(columns=['Family', 'Variety', 'File_Name', 'Avg_Entropy', 'Avg_Wordsize','Avg_Length',\
        'Avg_Num_Equals', 'Avg_Num_Dashes', 'Avg_Num_Slashes', 'Avg_Num_Pluses', 'Avg_Sum_RepChars', 'class'])
    excluded_files = pd.DataFrame(columns=['family', 'variety', 'file_name', 'class'])

    temp_dir = path.join(PROJECT_ROOT_DIR, 'temp-dir') 
    if not path.exists(temp_dir):
        makedirs(temp_dir)
    output_dir = temp_dir  # temp dir for extracting the apks
    apps_dir = temp_dir  # dummy variable

    amd_gt_df = pd.read_csv("AMD-ground-truth.csv")
    for _,row in amd_gt_df.iterrows():
        family = row['Family']
        variety = row['Variety']
        se_stat = row['String_Encryption_Stat']
        variety_files = file_tools.files_of_dir(path.join(amd_apks_folder, family, variety))
        for apk_file in variety_files:
            androdet_features_values = EFS.extract_features(
                path.join(amd_apks_folder, family, variety, apk_file),
                apps_dir, dexdump_dir, output_dir)
            if len(androdet_features_values) > 0:
                amd_androdet_dataset.loc[len(amd_androdet_dataset)] =\
                    [family, variety, apk_file[:-4]] + androdet_features_values + [se_stat]
            else:
                excluded_files.loc[len(excluded_files)] = [family, variety, apk_file[:-4], se_stat]
    amd_androdet_dataset_sorted = pd.DataFrame(
        columns=amd_androdet_dataset.columns) # make a sorted list of dataset for replication purposes
    amd_androdet_dataset = amd_androdet_dataset.groupby(['Family', 'Variety'])
    for _,family_variety_group in amd_androdet_dataset:
        amd_androdet_dataset_sorted = amd_androdet_dataset_sorted.append(
            family_variety_group.sort_values(by=['File_Name']))
    amd_androdet_dataset_sorted.to_csv('AMD-androdet-features.csv', index=False)
    excluded_files.to_csv('AMD-excluded-files.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="python code to extract the androdet feature values for the AMD dataset.\
        Run it in the amd-androdet folder using python 2.7\n\
        Note : As, for extracting the feature values, it calls the androdet EFS module, all the requirements of that module\
        including the dexdump disassembler should be installed")
    parser.add_argument('--amd_dataset', help="Folder containing the the amd dataset(The amd_data folder)")
    parser.add_argument('--dexdump_dir', help="Parent dir of the dexdump tool", default="/usr/bin")
    args = parser.parse_args()

    amd_apks_folder = args.amd_dataset
    dexdump_dir = args.dexdump_dir
    generate_amd_androdet_dataset(amd_apks_folder, dexdump_dir)
