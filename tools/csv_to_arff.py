
def csv2arff(data_file, relation_name):
    arff_file = data_file.replace("atm_", "")
    arff_file = arff_file.replace("csv", "arff")
    with open(arff_file, 'w') as f:
        f.writelines("@relation {}\n\n".format(relation_name))
        f.writelines("@attribute Avg_Entropy REAL\n")
        f.writelines("@attribute Avg_Wordsize REAL\n")
        f.writelines("@attribute Avg_Length REAL\n")
        f.writelines("@attribute Avg_Num_Equals REAL\n")
        f.writelines("@attribute Avg_Num_Dashes REAL\n")
        f.writelines("@attribute Avg_Num_Slashes REAL\n")
        f.writelines("@attribute Avg_Num_Pluses REAL\n")
        f.writelines("@attribute Avg_Sum_RepChars REAL\n")
        f.writelines("@attribute class REAL\n")
        f.writelines("@data\n")
        with open(data_file) as d_f:
            data_file_lines = d_f.readlines()
            for line in data_file_lines[1:]:
                f.writelines(line)
