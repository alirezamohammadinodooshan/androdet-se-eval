import pandas as pd
import sys
from os import path as path
from os import makedirs
import argparse
from tqdm import tqdm

PROJECT_ROOT_DIR = path.abspath('..')
sys.path.append(path.join(PROJECT_ROOT_DIR,'tools'))
import file_tools, EFS_with_returning_extracted_strings_no as EFS


def generate_praguard_androdet_dataset(praguard_string_encryption_apks_folder, dexdump_dir):
    pragaurd_androdet_dataset = pd.DataFrame(columns=['File_Name', 'Avg_Entropy', 'Avg_Wordsize', 'Avg_Length',\
        'Avg_Num_Equals', 'Avg_Num_Dashes', 'Avg_Num_Slashes', 'Avg_Num_Pluses', 'Avg_Sum_RepChars', 'No_of_Strings'])
    excluded_files = pd.DataFrame(columns=['file_name'])

    temp_dir = path.join(PROJECT_ROOT_DIR, 'temp-dir') 
    if not path.exists(temp_dir):
        makedirs(temp_dir)
    output_dir = temp_dir  # temp dir for extracting the apks

    praguard_apks = file_tools.full_path_of_files_of_dir_nested(praguard_string_encryption_apks_folder)
    for apk_file in tqdm(praguard_apks):
        androdet_features_values = EFS.extract_features(apk_file, output_dir, dexdump_dir, output_dir)
        if len(androdet_features_values) > 0:
            pragaurd_androdet_dataset.loc[len(pragaurd_androdet_dataset)] = \
            [apk_file.split('/')[-1].split('.')[0]]+androdet_features_values
        else:
            excluded_files.loc[len(excluded_files)] = [apk_file.split('/')[-1].split('.')[0]]
    pragaurd_androdet_dataset.sort_values(by=["File_Name"]).to_csv('pragaurd-androdet.csv', index=False)
    excluded_files.sort_values(by=["file_name"]).to_csv('pragaurd-excluded-files.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="python code to extract the androdet features for the praguard dataset\n\
        Run it in the praguard-androdet folder using python 2.7\n\
        Note : As, for extracting the features, it calls the androdet EFS module, all the requirements of androdet\
        including the dexdump disassembler should be installed")
    parser.add_argument('--dataset', help="Folder containing the the praguard se dataset(The STRING_ENCRYPTION_APK folder)")
    parser.add_argument('--dexdump_dir', help="Parent dir of the dexdump tool", default="/usr/bin")
    args = parser.parse_args()
    praguard_string_encryption_apks_folder = args.dataset
    dexdump_dir = args.dexdump_dir
    generate_praguard_androdet_dataset(praguard_string_encryption_apks_folder, dexdump_dir)
