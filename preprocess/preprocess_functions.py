import pandas as pd
import csv
import sys
from properties import data_folder_path

csv.field_size_limit(50000000)
     
def split_csv(csv_path):
    """
    Split the csv file into six folders based on the type of blocks

    :param csv_path
    :return:
    """

    # find the name of the csv file
    csv_name = csv_path.split('/')[-1]
    print(csv_name)

    type_of_changes = ['only_additions','only_deletions','both']

    data = pd.read_csv(csv_path, delimiter='\t', engine='python')

    data['code_additions_ast'].loc[(data['code_additions_ast'] == -1)] = '-1'
    data['code_deletions_ast'].loc[(data['code_deletions_ast'] == -1)] = '-1'
    data['code_additions'].loc[(data['code_additions'] == -1)] = '-1'
    data['code_deletions'].loc[(data['code_deletions'] == -1)] = '-1'
    
    # create a new data frame for the changes of type toc
    t_only_add_data = pd.DataFrame(columns=["sha", "filename","repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])
    t_only_del_data = pd.DataFrame(columns=["sha", "filename", "repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])
    t_both_data = pd.DataFrame(columns=["sha", "filename", "repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])

    for i, row in data.iterrows():
        if row['code_deletions'] == '-1' or row['code_deletions'] == -1 :
            temp = 'only_additions'
        elif row['code_additions'] == '-1' or row['code_additions'] == -1 :
            temp = 'only_deletions'
        else:
            temp = 'both'
        if temp == 'only_additions':
            t_only_add_data = t_only_add_data.append(row)
        elif temp == 'only_deletions':
            t_only_del_data = t_only_del_data.append(row)
        else:
            t_both_data = t_both_data.append(row)
        # save the data frame into a csv file to the appropriate folder
        
    t_only_add_data.reset_index(drop=True, inplace=True)
    t_only_add_data.to_csv(data_folder_path + '/data_split/' + 'only_additions/' + csv_name)
    
    t_only_del_data.reset_index(drop=True, inplace=True)
    t_only_del_data.to_csv(data_folder_path + '/data_split/' + 'only_deletions/' + csv_name)
    
    t_both_data.reset_index(drop=True, inplace=True)
    t_both_data.to_csv(data_folder_path + '/data_split/' + 'both/' + csv_name)