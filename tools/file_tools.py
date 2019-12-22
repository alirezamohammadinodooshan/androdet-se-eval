from os import listdir
from os.path import isfile, join, isdir


def files_of_dir(dir_name):
    file_names = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    return file_names


def full_path_of_files_of_dir(dir_name):
    file_names = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    return file_names


def full_path_of_files_of_dir_nested(dir_name):
    list_of_dir = listdir(dir_name)
    file_names = list()
    for file in list_of_dir:
        file_full_path = join(dir_name, file)
        if isdir(file_full_path):
            file_names = file_names + full_path_of_files_of_dir_nested(file_full_path)
        else:
            file_names.append(file_full_path)
    return file_names
