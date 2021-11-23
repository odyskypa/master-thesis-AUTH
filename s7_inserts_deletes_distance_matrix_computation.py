import os
from joblib import load,dump
from dataset_creation.distance_matrix import calculate_distance_matrix
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from properties import data_folder_path

def inserts_deletes_distance_matrix_computation(data_file_name, type_of_change):
    """
    Distance Matrix Computation per type of snippets

    :param data_file_name: the csv file of the snippets
    :param type_of_blocks
    """

    # load data with the blocks of code
    data_path = data_folder_path + "/preclustering/clusters/" + type_of_change + '/' + data_file_name
    data = pd.read_csv(data_path,index_col=0)
    print(data.shape)
    
    # remove suffix
    data_file_name_without_suffix = data_file_name.split('.')[0]
    
    if type_of_change == "only_additions":
        
        asts = data["code_inserts_ast"]
        
        distance_matrix_path = data_folder_path + "clustering/distance_matrices_inserts_deletes/" + type_of_change + "/" \
                           + data_file_name_without_suffix +'.joblib'
        
        if not os.path.exists(distance_matrix_path):

            distance_matrix = calculate_distance_matrix(asts)

            distance_matrix = distance_matrix + distance_matrix.T
            distance_matrix = squareform(distance_matrix)

            dump(distance_matrix, distance_matrix_path)
        else:
            distance_matrix = load(distance_matrix_path)

        distance_matrix = squareform(distance_matrix)

    elif type_of_change == "only_deletions":
        
        asts = data["code_deletes_ast"]
        
        distance_matrix_path = data_folder_path + "clustering/distance_matrices_inserts_deletes/" + type_of_change + "/" \
                           + data_file_name_without_suffix + '.joblib'
        
        if not os.path.exists(distance_matrix_path):

            distance_matrix = calculate_distance_matrix(asts)

            distance_matrix = distance_matrix + distance_matrix.T
            distance_matrix = squareform(distance_matrix)

            dump(distance_matrix, distance_matrix_path)
        else:
            distance_matrix = load(distance_matrix_path)

        distance_matrix = squareform(distance_matrix)
        
    else:
        
        add_asts = data["code_inserts_ast"]
        del_asts = data["code_deletes_ast"]
        
        add_distance_matrix_path = data_folder_path + "clustering/distance_matrices_inserts_deletes/" + type_of_change + "/" \
                           + data_file_name_without_suffix + "_additions.joblib"
        
        del_distance_matrix_path = data_folder_path + "clustering/distance_matrices_inserts_deletes/" + type_of_change + "/" \
                           + data_file_name_without_suffix + "_deletions.joblib"
        
        distance_matrix_path = data_folder_path + "clustering/distance_matrices_inserts_deletes/" + type_of_change + "/" \
                           + data_file_name_without_suffix + ".joblib"
                           
        if not os.path.exists(add_distance_matrix_path):

            add_distance_matrix = calculate_distance_matrix(add_asts)

            add_distance_matrix = add_distance_matrix + add_distance_matrix.T
            add_distance_matrix = squareform(add_distance_matrix)

            dump(add_distance_matrix, add_distance_matrix_path)
        else:
            add_distance_matrix = load(add_distance_matrix_path)

        add_distance_matrix = squareform(add_distance_matrix)
        
        if not os.path.exists(del_distance_matrix_path):

            del_distance_matrix = calculate_distance_matrix(del_asts)

            del_distance_matrix = del_distance_matrix + del_distance_matrix.T
            del_distance_matrix = squareform(del_distance_matrix)

            dump(del_distance_matrix, del_distance_matrix_path)
        else:
            del_distance_matrix = load(del_distance_matrix_path)

        del_distance_matrix = squareform(del_distance_matrix)
        
        if not os.path.exists(distance_matrix_path):
            
            distance_matrix = np.add(add_distance_matrix, del_distance_matrix)
            distance_matrix = squareform(distance_matrix)

            dump(distance_matrix, distance_matrix_path)
        else:
            distance_matrix = load(distance_matrix_path)
        
        distance_matrix = squareform(distance_matrix)
            
if __name__ == "__main__":
    
    type_of_changes = ['only_additions','only_deletions','both']
    for toc in type_of_changes:
        csv_names = []
        for csv_name in os.listdir(data_folder_path + 'preclustering/clusters/' + toc):
            csv_names.append(csv_name)
        for csv_name in csv_names:
            inserts_deletes_distance_matrix_computation(csv_name, toc)