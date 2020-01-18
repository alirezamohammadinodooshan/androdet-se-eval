# androdet-se-eval
*Codes for evaluating the AndrODet features for string encryption detection*

The repo consists of the following folders:
# 1- amd-androdet
### Goal
Extracting the AndrODet feature values for the AMD dataset
### How to use
The extracted feature values are currently in the amd-androdet folder. If you want to extract them again, just run gen-amd-androdet-features-values.py in its folder(amd-androdet) with **python 2.7** passing the following two arguments:
 * *--amd_dataset* : folder containing the AMD dataset apks
 * *--dexdump_dir* : parent dir of the dexdump tool
### Output
The following two files are generated:
 * *AMD-androdet-features*:  AndrODet feature values for the AMD dataset
 * *AMD-excluded-files*: AMD apks for which the androdet features could not be extracted
 > Note : The AMD-ground-truth file in this folder contains the ground truth for the string encryption status of each variety in the AMD dataset. It is used for generating the *AMD-androdet-features dataset*.

# 2- families-study
### Goal
To do the family-based study (train on all families except one) as mentioned in the comment paper
### How to use
Just run run-families-study.py in its folder(families-study) with **python 3** using the following arguments:
* *--dataset* : To generate the data set for each of the families (The training and testing sets)
* *--do_moa* : To do the moa analysis using the generated dataset for each family
* *--do_atm* : To do the atm analysis using the generated dataset for each family
### Output
More than the dataset folder, the following two files are generated:
 * *atm_results*:  tp,tn,fp,fn for each of the families evaluations(using  atm)
 * *moa_results*: tp,tn,fp,fn for each of the families evaluations(using moa)
> Note :As training the atm models may be time consuming, for each family, the best classifier selected by atm are currently imported to the repository. Delete them if you want to retrain the atm model of each family. If the best models are not deleted, they will be used for testing and the training procedure does not happen.


# 3- non-overlapping-study
### Goal
To do the nonoverlapping study as mentioned in the comment paper
### How to use
Just run run-non-overlapping-study.py in its folder(non-overlapping-study) with **python 3** using the following arguments:
* *--dataset* : To generate the data set for each of the 100 cross validations (The training and testing sets)
* *--do_moa* : To do the moa analysis using the generated dataset for each cross validation
* *--do_atm* : To do the atm analysis using the generated dataset for each validation
### Output
More than the dataset folder, the following two files are generated:
 * *atm_results*:  tp,tn,fp,fn for each of the cross validations(using  atm)
 * *moa_results*: tp,tn,fp,fn for each of the cross validations(using moa)
> Note : As training the atm models may be time consuming, for each cross validation, the best classifier selected by atm are currently imported to the repository. Delete them if you want to retrain the atm model of each cross validation. If the best models are not deleted, they will be used for testing and the training procedure does not happen.


# 4- praguard-androdet
### Goal
For extracting the AndrODet features values + number of strings in each apk from the PRAGuard dataset.

The extracted feature values are currently in praguard-androdet folder. If you want to extract them again, just run gen-praguard-androdet-features-values.py in its folder(praguard-androdet) with **python 2.7** passing the following two arguments:
 * *--dataset* : folder containing the "STRING_ENCRYPTION" apks of the praguard dataset
 * *--dexdump_dir* : parent dir of the dexdump tool
### Output
The following two files are generated:
 * *pragaurd-androdet*:  AndrODet feature values for the pragaurd dataset + number of strings in each file
 * *pragaurd-excluded-files*: paragaurd se apks for which the androdet features could not be extracted



# 5- random-study
### Goal
To do the random study as mentioned in the comment paper
### How to use
Just run run-random-study.py in its folder(random-study) with **python 3** using the following arguments:
* *--dataset* : To generate the data set for each of the 100 cross validations (The training and testing sets)
* *--do_moa* : To do the moa analysis using the generated dataset for each cross validation
* *--do_atm* : To do the atm analysis using the generated dataset for each validation
### Output
More than the dataset folder, the following two files are generated:
 * *atm_results*:  tp,tn,fp,fn for each of the cross validations(using  atm)
 * *moa_results*: tp,tn,fp,fn for each of the cross validations(using moa)
> Note : As training the atm models may be time consuming, for each cross validation, the best classifier selected by atm are currently imported to the repository. Delete them if you want to retrain the atm model of each cross validation. If the best models are not deleted, they will be used for testing and the training procedure does not happen.

# 6- reporting
### Goal
To extract the statistics mentioned in the comment paper
### How to use
Just run the jupyter notebook with a **python 3** kernel

# 7- tools
### Goal
Some tools referenced by the aforementioned parts, including the EFS module of the AndrODet work(accessed on Oct 2, 2019), and its modified version(EFS_with_returning_extracted_strings_no), which we made it to also extracts the number of strings in the apk. The tools folder also has the moa subfolder which includes the moa jar file and the eval.java file.  The goal of this file is to train a LeveragingBag moa classifier using a arff train file and return the  tp,tn,fp,fn of evaluting it on a arff test file( The train and test files are arguemnts which are passed from the families, non-overlapping and random studies)

# License
This repository is licensed under the GPL License. Consult the LICENSE.md file for more details

# Note
The results reported in the paper has been produced in  - the following environment:
 - Ubuntu (18.04 kernel 4.15)
 - Anaconda python
   (2.7 and 3.7)
 - Dexdump (8.1.0+r23-3~18.04)
 - Moa (release-2019.05.0)
 - OpenJDK Runtime Environment (build 1.8.0_232)
