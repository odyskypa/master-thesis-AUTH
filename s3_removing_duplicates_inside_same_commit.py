import numpy as np
import pandas as pd
import os
import datetime
from properties import data_folder_path
from dataset_creation.distance_matrix import pq_gram_distance, pq_grams_profiles, plot_distance_matrix, calculate_distance_matrix_same_commit, find_zero_distance_indexes
from tree_distance import parallel

def saving_df_to_csv(df1, indexes, col_indexes, final_path, start):
    
    # In this case we have zero distances between different asts so we keep
    # the modifications pointed by the indexes parameter.
    if len(indexes) > 0 and len(col_indexes) > 0:       
        keep=[]
        for index in indexes:
            keep_index = col_indexes.iloc[index]
            keep.append(keep_index)
        df = df1.loc[keep]
        del df["index"]
        if start == False:
            df.to_csv(final_path,sep = '\t', encoding = 'utf-8', index = False)
            start = True
        else:
            df.to_csv(final_path,sep= '\t', encoding = 'utf-8', index = False, header=None, mode="a")
    
    # This case is true when we have only one modification not in total, but in a single
    # type of modifications , e.g only_additions, only_deletions, without_minus_one.
    elif  len(indexes) == 0 and len(col_indexes) == 0:
        
        del df1["index"]
        if start == False:
            df1.to_csv(final_path, sep= '\t', encoding = 'utf-8', index = False)
            start = True
        else:
            df1.to_csv(final_path, sep= '\t', encoding = 'utf-8', index = False, header=None, mode="a")
    
    # In this case we have empty lists from find_zero_indexes so we keep all modifications
    # of the same commit.
    else:
        
        df = df1.loc[col_indexes]
        del df["index"]
        if start == False:
            df.to_csv(final_path, sep= '\t', encoding = 'utf-8', index = False)
            start = True
        else:
            df.to_csv(final_path, sep= '\t', encoding = 'utf-8', index = False, header=None, mode="a")
    return start

def remove_duplicates(commit_file):
    multiple_modifications_counter = 0
    single_modification_counter = 0
    total_only_add_counter = 0
    total_only_del_counter = 0
    total_both_counter = 0
    total_removed_duplicates_only_add_counter = 0
    total_removed_duplicates_only_del_counter = 0
    total_removed_duplicates_both_counter = 0
    
    data_file_name = commit_file
    data_path = data_folder_path + "commits_files/"  + data_file_name
    
    # Removing '_commits.csv' from string.
    data_file_name_without_suffix = data_file_name.split('.')[0][:-8]
    final_path = data_folder_path + "removed_duplicates_files/" + data_file_name_without_suffix + "_removed_duplicates.csv"
    
    """ if os.path.exists(final_path):
        print(final_path)
        continue """
    
    start = False
    
    # Reading commits.csv into data dataframe.
    data = pd.read_csv(data_path, delimiter='\t')
    
    
    data['code_additions_ast'].loc[(data['code_additions_ast'] == -1)] = '-1'
    data['code_deletions_ast'].loc[(data['code_deletions_ast'] == -1)] = '-1'
    data['code_additions'].loc[(data['code_additions'] == -1)] = '-1'
    data['code_deletions'].loc[(data['code_deletions'] == -1)] = '-1'
    
    # Picking the column 'sha' of commits. 
    shas = data["sha"]
    
    # For every unique commit_sha
    for sha in set(shas):
        
        # Picking as a dataframe only the elements with the same commit sha.
        sha_data_df = data[data.sha == sha]
        
        # Creating a column in the dataframe with the indexes of the rows.
        sha_data_df["index"] = sha_data_df.index
        
        # Only commits that process multiple files.
        if len(sha_data_df) > 1 :
            
            multiple_modifications_counter = multiple_modifications_counter + 1 
            # Creating dataframes with only the modifications of just additions or deletions.
            df_only_deletions = sha_data_df.loc[sha_data_df['code_additions'] == '-1']
            df_only_additions = sha_data_df.loc[sha_data_df['code_deletions'] == '-1']
            
            # Creating dataframe with both additions and deletions.
            df_both = sha_data_df.loc[(sha_data_df["code_additions"] != '-1') & (sha_data_df["code_deletions"] != '-1' )]      
            
            total_only_add_counter = total_only_add_counter + len(df_only_additions)
            total_only_del_counter = total_only_del_counter + len(df_only_deletions)
            total_both_counter = total_both_counter + len(df_both)
            
            # Keeping the indexes of all registrations for later use for all the three dataframes
            # above mentioned.                        
            only_deletions_col_indexes = df_only_deletions["index"]
            only_additions_col_indexes = df_only_additions["index"]
            both_col_indexes = df_both["index"]
            
            # Keeping the columns of code_additions_ast and code_deletions_ast in order
            # to calculate distance matrix for each column below for all three dfs.
            both_additions_asts = df_both["code_additions_ast"]      
            both_deletions_asts = df_both["code_deletions_ast"]
            
            only_additions_asts = df_only_additions["code_additions_ast"]      
                    
            only_deletions_asts = df_only_deletions["code_deletions_ast"]
            
            # Checking first the registrations that have both additions and deletions.
            if len(df_both) > 1 :                        
                
                # Seting saving filepath and calling function to calculate the distance matrix
                # for additions and deletions asts.
                try:                    
                    both_add_distance_matrix_path = data_folder_path + "distance_matrices/both_add_distance_matrices/" + data_file_name_without_suffix
                    both_add_distance_matrix = calculate_distance_matrix_same_commit(both_add_distance_matrix_path, both_additions_asts, sha)
                    #both_add_distance_matrix = 0.8 * both_add_distance_matrix
                    
                    both_del_distance_matrix_path = data_folder_path + "distance_matrices/both_del_distance_matrices/" + data_file_name_without_suffix
                    both_del_distance_matrix = calculate_distance_matrix_same_commit(both_del_distance_matrix_path, both_deletions_asts, sha)
                    #both_del_distance_matrix = 0.2 * both_del_distance_matrix
                    
                    both_final_distance_matrix_path = data_folder_path + "distance_matrices/both_final_distance_matrices/" + data_file_name_without_suffix + '/' + sha + '.txt'
                    if not os.path.exists(data_folder_path + "distance_matrices/both_final_distance_matrices/" + data_file_name_without_suffix + '/'):
                        os.makedirs(data_folder_path + "distance_matrices/both_final_distance_matrices/" + data_file_name_without_suffix + '/')
                    both_final_distance_matrix = np.add(both_add_distance_matrix, both_del_distance_matrix)
                    with open(both_final_distance_matrix_path,"w") as f:
                            f.write("\n".join(" ".join(map(str, x)) for x in both_final_distance_matrix))
                
                    # Creating a list with the indexes of the distance matrix with values equal to zero, so we
                    # could remove duplicates from the data dataframe.
                    zero_indexes = find_zero_distance_indexes(both_final_distance_matrix)
                            
                    # In this case we have zero distances and we keep only one from same additions - deletions.    
                    if len(zero_indexes) > 0 :
                    
                        scalar_indexes = list(range(len(df_both)))
                        # We remove the index of the all elements except of the first one in every tuple of indexes
                        # see find_zero_distance_indexes in dataset_creation.distance_matrix.
                        for i in range(len(zero_indexes)):
                            for j in range(len(zero_indexes[i])):
                                if j == max(range(len(zero_indexes[i]))):
                                    continue
                                if zero_indexes[i][j+1] in scalar_indexes:
                                    scalar_indexes.remove(zero_indexes[i][j+1])
                            
                        total_removed_duplicates_both_counter = total_removed_duplicates_both_counter + (len(both_final_distance_matrix) -len(scalar_indexes))
                        print(len(both_final_distance_matrix) -len(scalar_indexes), ' rows removed !')
                        start = saving_df_to_csv(df_both, scalar_indexes, both_col_indexes, final_path, start)
                    # In this case we dont have zero distances so we keep all the files.            
                    else:
                        start = saving_df_to_csv(df_both, [], both_col_indexes, final_path, start)                  
                except Exception as e:
                    print(e)
            # In this case we only have one file modification in the same commit and we keep it aswell.
            elif len(df_both) == 1:
                start = saving_df_to_csv(df_both, [], [], final_path, start)
                
            # Now doing the same for commits with values for deletions equal to minus one.
            if len(df_only_additions) > 1 :
                
                try:
                    only_add_distance_matrix_path = data_folder_path + "distance_matrices/only_add_distance_matrices/" + data_file_name_without_suffix
                    only_add_distance_matrix = calculate_distance_matrix_same_commit(only_add_distance_matrix_path, only_additions_asts, sha)
                
                    add_scalar_indexes = list()
                    add_zero_indexes = find_zero_distance_indexes(only_add_distance_matrix)
                
                    if len(add_zero_indexes) > 0:
                    
                        add_scalar_indexes = list(range(len(df_only_additions)))
                        
                        # We remove the index of the all elements except of the first one in every tuple of indexes
                        # see find_zero_distance_indexes in dataset_creation.distance_matrix.
                        for i in range(len(add_zero_indexes)):
                            for j in range(len(add_zero_indexes[i])):
                                if j == max(range(len(add_zero_indexes[i]))):
                                    continue
                                if add_zero_indexes[i][j+1] in add_scalar_indexes:
                                    add_scalar_indexes.remove(add_zero_indexes[i][j+1])
                        
                        total_removed_duplicates_only_add_counter = total_removed_duplicates_only_add_counter + (len(only_add_distance_matrix) - len(add_scalar_indexes))            
                        print(len(only_add_distance_matrix) - len(add_scalar_indexes), ' rows removed !')
                        start = saving_df_to_csv(df_only_additions, add_scalar_indexes, only_additions_col_indexes, final_path, start)
                    else:
                        start = saving_df_to_csv(df_only_additions, [], only_additions_col_indexes, final_path, start)
                except Exception as e:
                    print(e)
                    
            elif len(df_only_additions) == 1 :
                
                start = saving_df_to_csv(df_only_additions, [], [], final_path, start)
            # Now doing the same for commits with values for deletions equal to minus one.
            if len(df_only_deletions) > 1 :
                
                try:
                    only_del_distance_matrix_path = data_folder_path + "distance_matrices/only_del_distance_matrices/" + data_file_name_without_suffix
                    only_del_distance_matrix = calculate_distance_matrix_same_commit(only_del_distance_matrix_path, only_deletions_asts, sha)
                
                    del_scalar_indexes = list()
                    del_zero_indexes = find_zero_distance_indexes(only_del_distance_matrix)
                
                
                    if len(del_zero_indexes) > 0:
                    
                        del_scalar_indexes = list(range(len(df_only_deletions)))
                        
                        # We remove the index of the all elements except of the first one in every tuple of indexes
                        # see find_zero_distance_indexes in dataset_creation.distance_matrix.
                        for i in range(len(del_zero_indexes)):
                            for j in range(len(del_zero_indexes[i])):
                                if j == max(range(len(del_zero_indexes[i]))):
                                    continue
                                if del_zero_indexes[i][j+1] in del_scalar_indexes:
                                    del_scalar_indexes.remove(del_zero_indexes[i][j+1])
                        
                        total_removed_duplicates_only_del_counter = total_removed_duplicates_only_del_counter + (len(only_del_distance_matrix) - len(del_scalar_indexes))
                        print(len(only_del_distance_matrix) - len(del_scalar_indexes), ' rows removed !')
                        start = saving_df_to_csv(df_only_deletions, del_scalar_indexes, only_deletions_col_indexes, final_path, start)
                        
                    else:
                        start = saving_df_to_csv(df_only_deletions, [], only_deletions_col_indexes, final_path, start)
                        
                except Exception as e:
                    print(e)
            
            elif len(df_only_deletions) == 1 :
                
                start = saving_df_to_csv(df_only_deletions, [], [], final_path, start)
        
        # Commits that process a single file.
        else:
            single_modification_counter = single_modification_counter + 1 
            del sha_data_df["index"]
            if start == False:
                sha_data_df.to_csv(final_path, sep='\t', encoding='utf-8', index = False)
                start = True
            else:
                sha_data_df.to_csv(final_path, sep= '\t', encoding = 'utf-8', index = False, header=None, mode="a")

    counters = {'multiple_modifications_counter':multiple_modifications_counter,\
                'single_modification_counter':single_modification_counter,\
                'total_only_add_counter': total_only_add_counter,\
                'total_only_del_counter':total_only_del_counter,\
                'total_both_counter':total_both_counter,\
                'total_removed_duplicates_only_add_counter':total_removed_duplicates_only_add_counter,\
                'total_removed_duplicates_only_del_counter':total_removed_duplicates_only_del_counter,\
                'total_removed_duplicates_both_counter':total_removed_duplicates_both_counter}
    df = pd.DataFrame(counters, index=[data_file_name_without_suffix])
    
    print('Repo Done')
    return df
    

if __name__ == '__main__':
    
    # Creating the path for removed_duplicates_files
    if not os.path.exists(data_folder_path + "removed_duplicates_files/"):
        os.makedirs(data_folder_path + "removed_duplicates_files/")
        
    if not os.path.exists(data_folder_path + "distance_matrices/both_final_distance_matrices/"):
        os.makedirs(data_folder_path + "distance_matrices/both_final_distance_matrices/")
    
    counters_path = os.path.join(data_folder_path + "s3_counters.csv")    

    begin_time = datetime.datetime.now()
    commit_files = [commit_file for commit_file in os.listdir(data_folder_path + "commits_files/") if commit_file.endswith("commits.csv")]
    start0 = False
        
    for commit_file in commit_files:
        df = remove_duplicates(commit_file)
        if start0 == False:
            df.to_csv(counters_path)
            start0 = True
        else:
            df.to_csv(counters_path, header=None, mode="a")
            
    print("Time for s3 to be completed: ",datetime.datetime.now() - begin_time)    
    print("Done !!!")
            
        
        