import os
import sys
import csv
import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from properties import data_folder_path
from tree_distance import parallel
from preprocess.code_complexity import *
from preprocess.preprocess_functions import split_csv
plt.style.use('fivethirtyeight')
sns.set()

def data_split(folder_with_csv,type_of_changes):
    """
    The data_split function split the dataset with csv files into three folders based on the type of changes.
    More specifically it splits in only_additions,only_deletions,both.

    :param folder_with_csv: the path of the folder with the csv files
    :param : type_of_changes: ["only_additions","only_deletions","both"]
    :return:
    """
    if not os.path.exists(data_folder_path + '/data_split/'):
        os.makedirs(data_folder_path + '/data_split/')
        for toc in type_of_changes:
            os.makedirs(data_folder_path + '/data_split/' + toc)

    print("Data split:")

    # find all the csv into the folder
    csv_paths = []

    for csv_name in os.listdir(folder_with_csv):
        csv_path = os.path.join(folder_with_csv, csv_name)

        csv_paths.append(csv_path)

    # We make use of multiprocessing library in order to split the dataset in parallel
    # Each worker calls the function split csv so to separate the blocks based on the type
    parallel.pool_of_workers(4, split_csv, csv_paths)

def merge_changes(csv_files_path, type_of_changes):
    """
    Merge the files with the changes of code into one csv file for every type of it:
    only_additions,only_deletions,both.

    :param type_of_changes:
    :param csv_files_path:
    :return:
    """

    if not os.path.exists(data_folder_path + '/merged_changes/'):
        os.makedirs(data_folder_path + '/merged_changes/')
    if not os.path.exists(data_folder_path + '/merged_changes/' + type_of_changes + '/'):
        os.makedirs(data_folder_path + '/merged_changes/' + type_of_changes + '/')

    paths_csv_files = [os.path.join(csv_files_path, csv_file) for csv_file in os.listdir(csv_files_path)]

    csv_name = data_folder_path + '/merged_changes/' + str(type_of_changes) + "/" + str(type_of_changes) + "_changes.csv"
    header = True

    for csv_file in paths_csv_files:

        print(csv_file)
        temp_data = pd.read_csv(csv_file, engine='python',index_col=[0])
        temp_data.to_csv(csv_name, header=header, mode='a',index = False)
        header = False
    temp_data = pd.read_csv(csv_name, engine='python')
    temp_data.reset_index(drop=True, inplace=True)
    temp_data.to_csv(csv_name)

def figures_directories_creation(type_of_changes):
        
    folders_names = ["correlation",'boxplots','histogramms']
    if not os.path.exists(data_folder_path + 'figures/'):
        os.makedirs(data_folder_path + 'figures/')

    for folder in folders_names:
        if not os.path.exists(data_folder_path + 'figures/' + folder + '/'):
            os.makedirs(data_folder_path + 'figures/' + folder + '/')

def features_calculation(type_of_change):
    
    data_path = data_folder_path + "merged_changes/"  + type_of_change + '/' + type_of_change + '_changes.csv'
    
    print("Calculating features for " + type_of_change + " changes.")
    
    data = pd.read_csv(data_path)
    data_len = len(data)
    
    if not os.path.exists(data_folder_path + "features/"):
        os.makedirs(data_folder_path + "features/")
    
    features_path = data_folder_path + "features/"  + type_of_change + '_features.csv'
    
    if type_of_change == 'only_additions':
        
        number_of_lines = []
        number_of_AST_children = []
        cc = []
        number_of_names = []
        unique_operators_count = []
        tob_flags = []
        number_of_method_invocations = []
        number_of_if_statements = []
        number_of_variable_statements = []
        number_of_switch_statements = []
        number_of_conditional_expression = []
        number_of_parenthesized_expression = []
        number_of_class_instance_creations = []
        number_of_while_statement = []
        number_of_for_statement = []
        number_of_try_statement = []
        number_of_return_statement = []
        number_of_single_variable_delcaration = []
        number_of_qualified_name = []
        
        for code_addition,code_addition_ast \
                    in zip(data['code_additions'],data['code_inserts_ast']):
                        
            number_of_lines.append(get_number_of_lines_changed(code_addition))
            number_of_AST_children.append(get_number_of_children(code_addition_ast))
            number_of_names.append(get_number_of_names(code_addition_ast))
            cc.append(cyclomatic_complexity(code_addition))
            tob_flags.append(contains_type_of_block(code_addition))
            unique_operators_count.append(get_unique_operators_count(code_addition))
            number_of_method_invocations.append(get_number_of_method_invocations(code_addition_ast))
            number_of_if_statements.append(get_number_of_if_statements(code_addition_ast))
            number_of_variable_statements.append(get_number_of_variable_statements(code_addition_ast))
            number_of_switch_statements.append(get_number_of_switch_statements(code_addition_ast))
            number_of_conditional_expression.append(get_number_of_conditional_expression(code_addition_ast))
            number_of_parenthesized_expression.append(get_number_of_parenthesized_expression(code_addition_ast))
            number_of_class_instance_creations.append(get_number_of_class_instance_creations(code_addition_ast))
            number_of_while_statement.append(get_number_of_while_statement(code_addition_ast))
            number_of_for_statement.append(get_number_of_for_statement(code_addition_ast))
            number_of_try_statement.append(get_number_of_try_statement(code_addition_ast))
            number_of_return_statement.append(get_number_of_return_statement(code_addition_ast))
            number_of_single_variable_delcaration.append(get_number_of_single_variable_delcaration(code_addition_ast))
            number_of_qualified_name.append(get_number_of_qualified_name(code_addition_ast))
        
        np_number_of_lines = np.array(number_of_lines)
        np_number_of_AST_children = np.array(number_of_AST_children)
        np_number_of_names = np.array(number_of_names)
        np_cc = np.array(cc)
        np_tob_flags = np.array(tob_flags)
        np_unique_operators_count = np.array(unique_operators_count)
        np_number_of_method_invocations = np.array(number_of_method_invocations)
        np_number_of_if_statements = np.array(number_of_if_statements)
        np_number_of_variable_statements = np.array(number_of_variable_statements)
        np_number_of_switch_statements = np.array(number_of_switch_statements)
        np_number_of_conditional_expression = np.array(number_of_conditional_expression)
        np_number_of_parenthesized_expression = np.array(number_of_parenthesized_expression)
        np_number_of_class_instance_creations = np.array(number_of_class_instance_creations)
        np_number_of_while_statement = np.array(number_of_while_statement)
        np_number_of_for_statement = np.array(number_of_for_statement)
        np_number_of_try_statement = np.array(number_of_try_statement)
        np_number_of_return_statement = np.array(number_of_return_statement)
        np_number_of_single_variable_delcaration = np.array(number_of_single_variable_delcaration)
        np_number_of_qualified_name = np.array(number_of_qualified_name)
        for_flag = np_tob_flags[:,0]
        while_flag = np_tob_flags[:,1]
        do_flag = np_tob_flags[:,2]
        switch_flag = np_tob_flags[:,3]
        if_flag = np_tob_flags[:,4]
        try_flag = np_tob_flags[:,5]
        
        d = {'number_of_lines': np_number_of_lines,
             'number_of_AST_children': np_number_of_AST_children,
             'number_of_names':np_number_of_names,
             'cc':np_cc,
             'unique_operators_count':np_unique_operators_count,
             'number_of_method_invocations' : np_number_of_method_invocations,
             'number_of_if_statements': np_number_of_if_statements,
             'number_of_variable_statements': np_number_of_variable_statements,
             'number_of_switch_statements':np_number_of_switch_statements,
             'number_of_conditional_expression': np_number_of_conditional_expression,
             'number_of_parenthesized_expression':np_number_of_parenthesized_expression,
             'number_of_class_instance_creations':np_number_of_class_instance_creations,
             'number_of_while_statement':np_number_of_while_statement,
             'number_of_for_statement':np_number_of_for_statement,
             'number_of_try_statement':np_number_of_try_statement,
             'number_of_return_statement':np_number_of_return_statement,
             'number_of_single_variable_delcaration':np_number_of_single_variable_delcaration,
             'number_of_qualified_name':np_number_of_qualified_name,
             'for_flag':for_flag,
             'while_flag':while_flag,
             'do_flag':do_flag,
             'switch_flag':switch_flag,
             'if_flag':if_flag,
             'try_flag':try_flag
        }
    
    elif type_of_change == 'only_deletions':
        
        number_of_lines = []
        number_of_AST_children = []
        cc = []
        number_of_names = []
        unique_operators_count = []
        tob_flags = []
        number_of_method_invocations = []
        number_of_if_statements = []
        number_of_variable_statements = []
        number_of_switch_statements = []
        number_of_conditional_expression = []
        number_of_parenthesized_expression = []
        number_of_class_instance_creations = []
        number_of_while_statement = []
        number_of_for_statement = []
        number_of_try_statement = []
        number_of_return_statement = []
        number_of_single_variable_delcaration = []
        number_of_qualified_name = []
        
        for code_deletion,code_deletion_ast, \
                    in zip(data['code_deletions'],data['code_deletes_ast']):
            number_of_lines.append(get_number_of_lines_changed(code_deletion))
            number_of_AST_children.append(get_number_of_children(code_deletion_ast))
            number_of_names.append(get_number_of_names(code_deletion_ast))
            cc.append(cyclomatic_complexity(code_deletion))
            tob_flags.append(contains_type_of_block(code_deletion))
            unique_operators_count.append(get_unique_operators_count(code_deletion))
            number_of_method_invocations.append(get_number_of_method_invocations(code_deletion_ast))
            number_of_if_statements.append(get_number_of_if_statements(code_deletion_ast))
            number_of_variable_statements.append(get_number_of_variable_statements(code_deletion_ast))
            number_of_switch_statements.append(get_number_of_switch_statements(code_deletion_ast))
            number_of_conditional_expression.append(get_number_of_conditional_expression(code_deletion_ast))
            number_of_parenthesized_expression.append(get_number_of_parenthesized_expression(code_deletion_ast))
            number_of_class_instance_creations.append(get_number_of_class_instance_creations(code_deletion_ast))
            number_of_while_statement.append(get_number_of_while_statement(code_deletion_ast))
            number_of_for_statement.append(get_number_of_for_statement(code_deletion_ast))
            number_of_try_statement.append(get_number_of_try_statement(code_deletion_ast))
            number_of_return_statement.append(get_number_of_return_statement(code_deletion_ast))
            number_of_single_variable_delcaration.append(get_number_of_single_variable_delcaration(code_deletion_ast))
            number_of_qualified_name.append(get_number_of_qualified_name(code_deletion_ast))
            
            
        np_number_of_lines = np.array(number_of_lines)
        np_number_of_AST_children = np.array(number_of_AST_children)
        np_number_of_names = np.array(number_of_names)
        np_cc = np.array(cc)
        np_tob_flags = np.array(tob_flags)
        np_unique_operators_count = np.array(unique_operators_count)
        np_number_of_method_invocations = np.array(number_of_method_invocations)
        np_number_of_if_statements = np.array(number_of_if_statements)
        np_number_of_variable_statements = np.array(number_of_variable_statements)
        np_number_of_switch_statements = np.array(number_of_switch_statements)
        np_number_of_conditional_expression = np.array(number_of_conditional_expression)
        np_number_of_parenthesized_expression = np.array(number_of_parenthesized_expression)
        np_number_of_class_instance_creations = np.array(number_of_class_instance_creations)
        np_number_of_while_statement = np.array(number_of_while_statement)
        np_number_of_for_statement = np.array(number_of_for_statement)
        np_number_of_try_statement = np.array(number_of_try_statement)
        np_number_of_return_statement = np.array(number_of_return_statement)
        np_number_of_single_variable_delcaration = np.array(number_of_single_variable_delcaration)
        np_number_of_qualified_name = np.array(number_of_qualified_name)
        for_flag = np_tob_flags[:,0]
        while_flag = np_tob_flags[:,1]
        do_flag = np_tob_flags[:,2]
        switch_flag = np_tob_flags[:,3]
        if_flag = np_tob_flags[:,4]
        try_flag = np_tob_flags[:,5]
        
        d = {'number_of_lines': np_number_of_lines,
             'number_of_AST_children': np_number_of_AST_children,
             'number_of_names':np_number_of_names,
             'cc':np_cc,
             'unique_operators_count':np_unique_operators_count,
             'number_of_method_invocations' : np_number_of_method_invocations,
             'number_of_if_statements': np_number_of_if_statements,
             'number_of_variable_statements': np_number_of_variable_statements,
             'number_of_switch_statements':np_number_of_switch_statements,
             'number_of_conditional_expression': np_number_of_conditional_expression,
             'number_of_parenthesized_expression':np_number_of_parenthesized_expression,
             'number_of_class_instance_creations':np_number_of_class_instance_creations,
             'number_of_while_statement':np_number_of_while_statement,
             'number_of_for_statement':np_number_of_for_statement,
             'number_of_try_statement':np_number_of_try_statement,
             'number_of_return_statement':np_number_of_return_statement,
             'number_of_single_variable_delcaration':np_number_of_single_variable_delcaration,
             'number_of_qualified_name':np_number_of_qualified_name,
             'for_flag':for_flag,
             'while_flag':while_flag,
             'do_flag':do_flag,
             'switch_flag':switch_flag,
             'if_flag':if_flag,
             'try_flag':try_flag
        }
        
    else:
        
        add_number_of_lines = []
        add_number_of_AST_children = []
        add_cc = []
        del_cc = []
        add_number_of_names = []
        add_unique_operators_count = []
        add_tob_flags = []
        add_number_of_method_invocations = []
        add_number_of_if_statements = []
        add_number_of_variable_statements = []
        add_number_of_switch_statements = []
        add_number_of_conditional_expression = []
        add_number_of_parenthesized_expression = []
        add_number_of_class_instance_creations = []
        add_number_of_while_statement = []
        add_number_of_for_statement = []
        add_number_of_try_statement = []
        add_number_of_return_statement = []
        add_number_of_single_variable_delcaration = []
        add_number_of_qualified_name = []
        del_number_of_lines = []
        del_number_of_AST_children = []
        del_cc_before = []
        del_cc_after = []
        del_number_of_names = []
        del_unique_operators_count = []
        del_tob_flags = []
        del_number_of_method_invocations = []
        del_number_of_if_statements = []
        del_number_of_variable_statements = []
        del_number_of_switch_statements = []
        del_number_of_conditional_expression = []
        del_number_of_parenthesized_expression = []
        del_number_of_class_instance_creations = []
        del_number_of_while_statement = []
        del_number_of_for_statement = []
        del_number_of_try_statement = []
        del_number_of_return_statement = []
        del_number_of_single_variable_delcaration = []
        del_number_of_qualified_name = []
        
        
        for code_deletion,code_addition,code_deletion_ast,code_addition_ast \
                    in zip(data['code_deletions'],data['code_additions'],data['code_deletes_ast'],data['code_inserts_ast']):
            
            add_number_of_lines.append(get_number_of_lines_changed(code_addition))
            add_number_of_AST_children.append(get_number_of_children(code_addition_ast))
            add_number_of_names.append(get_number_of_names(code_addition_ast))
            add_cc.append(cyclomatic_complexity(code_addition))
            add_tob_flags.append(contains_type_of_block(code_addition))
            add_unique_operators_count.append(get_unique_operators_count(code_addition))
            add_number_of_method_invocations.append(get_number_of_method_invocations(code_addition_ast))
            add_number_of_if_statements.append(get_number_of_if_statements(code_addition_ast))
            add_number_of_variable_statements.append(get_number_of_variable_statements(code_addition_ast))
            add_number_of_switch_statements.append(get_number_of_switch_statements(code_addition_ast))
            add_number_of_conditional_expression.append(get_number_of_conditional_expression(code_addition_ast))
            add_number_of_parenthesized_expression.append(get_number_of_parenthesized_expression(code_addition_ast))
            add_number_of_class_instance_creations.append(get_number_of_class_instance_creations(code_addition_ast))
            add_number_of_while_statement.append(get_number_of_while_statement(code_addition_ast))
            add_number_of_for_statement.append(get_number_of_for_statement(code_addition_ast))
            add_number_of_try_statement.append(get_number_of_try_statement(code_addition_ast))
            add_number_of_return_statement.append(get_number_of_return_statement(code_addition_ast))
            add_number_of_single_variable_delcaration.append(get_number_of_single_variable_delcaration(code_addition_ast))
            add_number_of_qualified_name.append(get_number_of_qualified_name(code_addition_ast))
            
            del_number_of_lines.append(get_number_of_lines_changed(code_deletion))
            del_number_of_AST_children.append(get_number_of_children(code_deletion_ast))
            del_number_of_names.append(get_number_of_names(code_deletion_ast))
            del_cc.append(cyclomatic_complexity(code_deletion))
            del_tob_flags.append(contains_type_of_block(code_deletion))
            del_unique_operators_count.append(get_unique_operators_count(code_deletion))
            del_number_of_method_invocations.append(get_number_of_method_invocations(code_deletion_ast))
            del_number_of_if_statements.append(get_number_of_if_statements(code_deletion_ast))
            del_number_of_variable_statements.append(get_number_of_variable_statements(code_deletion_ast))
            del_number_of_switch_statements.append(get_number_of_switch_statements(code_deletion_ast))
            del_number_of_conditional_expression.append(get_number_of_conditional_expression(code_deletion_ast))
            del_number_of_parenthesized_expression.append(get_number_of_parenthesized_expression(code_deletion_ast))
            del_number_of_class_instance_creations.append(get_number_of_class_instance_creations(code_deletion_ast))
            del_number_of_while_statement.append(get_number_of_while_statement(code_deletion_ast))
            del_number_of_for_statement.append(get_number_of_for_statement(code_deletion_ast))
            del_number_of_try_statement.append(get_number_of_try_statement(code_deletion_ast))
            del_number_of_return_statement.append(get_number_of_return_statement(code_deletion_ast))
            del_number_of_single_variable_delcaration.append(get_number_of_single_variable_delcaration(code_deletion_ast))
            del_number_of_qualified_name.append(get_number_of_qualified_name(code_deletion_ast))
        
        np_add_number_of_lines = np.array(add_number_of_lines)
        np_add_number_of_AST_children = np.array(add_number_of_AST_children)
        np_add_number_of_names = np.array(add_number_of_names)
        np_add_cc = np.array(add_cc)
        np_add_tob_flags = np.array(add_tob_flags)
        np_add_unique_operators_count = np.array(add_unique_operators_count)
        np_add_number_of_method_invocations = np.array(add_number_of_method_invocations)
        np_add_number_of_if_statements = np.array(add_number_of_if_statements)
        np_add_number_of_variable_statements = np.array(add_number_of_variable_statements)
        np_add_number_of_switch_statements = np.array(add_number_of_switch_statements)
        np_add_number_of_conditional_expression = np.array(add_number_of_conditional_expression)
        np_add_number_of_parenthesized_expression = np.array(add_number_of_parenthesized_expression)
        np_add_number_of_class_instance_creations = np.array(add_number_of_class_instance_creations)
        np_add_number_of_while_statement = np.array(add_number_of_while_statement)
        np_add_number_of_for_statement = np.array(add_number_of_for_statement)
        np_add_number_of_try_statement = np.array(add_number_of_try_statement)
        np_add_number_of_return_statement = np.array(add_number_of_return_statement)
        np_add_number_of_single_variable_delcaration = np.array(add_number_of_single_variable_delcaration)
        np_add_number_of_qualified_name = np.array(add_number_of_qualified_name)
        add_for_flag = np_add_tob_flags[:,0]
        add_while_flag = np_add_tob_flags[:,1]
        add_do_flag = np_add_tob_flags[:,2]
        add_switch_flag = np_add_tob_flags[:,3]
        add_if_flag = np_add_tob_flags[:,4]
        add_try_flag = np_add_tob_flags[:,5]
        
        np_del_number_of_lines = np.array(del_number_of_lines)
        np_del_number_of_AST_children = np.array(del_number_of_AST_children)
        np_del_number_of_names = np.array(del_number_of_names)
        np_del_cc = np.array(del_cc)
        np_del_tob_flags = np.array(del_tob_flags)
        np_del_unique_operators_count = np.array(del_unique_operators_count)
        np_del_number_of_method_invocations = np.array(del_number_of_method_invocations)
        np_del_number_of_if_statements = np.array(del_number_of_if_statements)
        np_del_number_of_variable_statements = np.array(del_number_of_variable_statements)
        np_del_number_of_switch_statements = np.array(del_number_of_switch_statements)
        np_del_number_of_conditional_expression = np.array(del_number_of_conditional_expression)
        np_del_number_of_parenthesized_expression = np.array(del_number_of_parenthesized_expression)
        np_del_number_of_class_instance_creations = np.array(del_number_of_class_instance_creations)
        np_del_number_of_while_statement = np.array(del_number_of_while_statement)
        np_del_number_of_for_statement = np.array(del_number_of_for_statement)
        np_del_number_of_try_statement = np.array(del_number_of_try_statement)
        np_del_number_of_return_statement = np.array(del_number_of_return_statement)
        np_del_number_of_single_variable_delcaration = np.array(del_number_of_single_variable_delcaration)
        np_del_number_of_qualified_name = np.array(del_number_of_qualified_name)
        del_for_flag = np_del_tob_flags[:,0]
        del_while_flag = np_del_tob_flags[:,1]
        del_do_flag = np_del_tob_flags[:,2]
        del_switch_flag = np_del_tob_flags[:,3]
        del_if_flag = np_del_tob_flags[:,4]
        del_try_flag = np_del_tob_flags[:,5]
        
        d = {'add_number_of_lines': np_add_number_of_lines,
             'add_number_of_AST_children': np_add_number_of_AST_children,
             'add_number_of_names':np_add_number_of_names,
             'add_cc':np_add_cc,
             'add_unique_operators_count':np_add_unique_operators_count,
             'add_number_of_method_invocations' : np_add_number_of_method_invocations,
             'add_number_of_if_statements': np_add_number_of_if_statements,
             'add_number_of_variable_statements': np_add_number_of_variable_statements,
             'add_number_of_switch_statements':np_add_number_of_switch_statements,
             'add_number_of_conditional_expression': np_add_number_of_conditional_expression,
             'add_number_of_parenthesized_expression':np_add_number_of_parenthesized_expression,
             'add_number_of_class_instance_creations':np_add_number_of_class_instance_creations,
             'add_number_of_while_statement':np_add_number_of_while_statement,
             'add_number_of_for_statement':np_add_number_of_for_statement,
             'add_number_of_try_statement':np_add_number_of_try_statement,
             'add_number_of_return_statement':np_add_number_of_return_statement,
             'add_number_of_single_variable_delcaration':np_add_number_of_single_variable_delcaration,
             'add_number_of_qualified_name':np_add_number_of_qualified_name,
             'add_for_flag':add_for_flag,
             'add_while_flag':add_while_flag,
             'add_do_flag':add_do_flag,
             'add_switch_flag':add_switch_flag,
             'add_if_flag':add_if_flag,
             'add_try_flag':add_try_flag,
             'del_number_of_lines': np_del_number_of_lines,
             'del_number_of_AST_children': np_del_number_of_AST_children,
             'del_number_of_names':np_del_number_of_names,
             'del_cc':np_del_cc,
             'del_unique_operators_count':np_del_unique_operators_count,
             'del_number_of_method_invocations' : np_del_number_of_method_invocations,
             'del_number_of_if_statements': np_del_number_of_if_statements,
             'del_number_of_variable_statements': np_del_number_of_variable_statements,
             'del_number_of_switch_statements':np_del_number_of_switch_statements,
             'del_number_of_conditional_expression': np_del_number_of_conditional_expression,
             'del_number_of_parenthesized_expression':np_del_number_of_parenthesized_expression,
             'del_number_of_class_instance_creations':np_del_number_of_class_instance_creations,
             'del_number_of_while_statement':np_del_number_of_while_statement,
             'del_number_of_for_statement':np_del_number_of_for_statement,
             'del_number_of_try_statement':np_del_number_of_try_statement,
             'del_number_of_return_statement':np_del_number_of_return_statement,
             'del_number_of_single_variable_delcaration':np_del_number_of_single_variable_delcaration,
             'del_number_of_qualified_name':np_del_number_of_qualified_name,
             'del_for_flag':del_for_flag,
             'del_while_flag':del_while_flag,
             'del_do_flag':del_do_flag,
             'del_switch_flag':del_switch_flag,
             'del_if_flag':del_if_flag,
             'del_try_flag':del_try_flag
        }
    
    features = pd.DataFrame(d)
    
    if type_of_change == 'both':
        
        features['cc'] = features['add_cc'] + features['del_cc']
        features = features.drop(columns=['add_cc','del_cc'])

        features['number_of_lines'] = features['add_number_of_lines'] + features['del_number_of_lines']
        features = features.drop(columns=['add_number_of_lines','del_number_of_lines'])
        
        features['number_of_AST_children'] = features['add_number_of_AST_children'] + features['del_number_of_AST_children']
        features = features.drop(columns=['add_number_of_AST_children','del_number_of_AST_children'])

        features['number_of_names'] = features['add_number_of_names'] + features['del_number_of_names']
        features = features.drop(columns=['add_number_of_names','del_number_of_names'])

        features['unique_operators'] = features['add_unique_operators_count'] + features['del_unique_operators_count']
        features = features.drop(columns=['add_unique_operators_count','del_unique_operators_count'])

        features['number_of_method_invocations'] = features['add_number_of_method_invocations'] + features['del_number_of_method_invocations']
        features = features.drop(columns=['add_number_of_method_invocations','del_number_of_method_invocations'])

        features['number_of_if_statements'] = features['add_number_of_if_statements'] + features['del_number_of_if_statements']
        features = features.drop(columns=['add_number_of_if_statements','del_number_of_if_statements'])

        features['number_of_variable_statements'] = features['add_number_of_variable_statements'] + features['del_number_of_variable_statements']
        features = features.drop(columns=['add_number_of_variable_statements','del_number_of_variable_statements'])

        features['number_of_switch_statements'] = features['add_number_of_switch_statements'] + features['del_number_of_switch_statements']
        features = features.drop(columns=['add_number_of_switch_statements','del_number_of_switch_statements'])

        features['number_of_conditional_expression'] = features['add_number_of_conditional_expression'] + features['del_number_of_conditional_expression']
        features = features.drop(columns=['add_number_of_conditional_expression','del_number_of_conditional_expression'])

        features['number_of_parenthesized_expression'] = features['add_number_of_parenthesized_expression'] + features['del_number_of_parenthesized_expression']
        features = features.drop(columns=['add_number_of_parenthesized_expression','del_number_of_parenthesized_expression'])

        features['number_of_class_instance_creations'] = features['add_number_of_class_instance_creations'] + features['del_number_of_class_instance_creations']
        features = features.drop(columns=['add_number_of_class_instance_creations','del_number_of_class_instance_creations'])

        features['number_of_while_statement'] = features['add_number_of_while_statement'] + features['del_number_of_while_statement']
        features = features.drop(columns=['add_number_of_while_statement','del_number_of_while_statement'])

        features['number_of_for_statement'] = features['add_number_of_for_statement'] + features['del_number_of_for_statement']
        features = features.drop(columns=['add_number_of_for_statement','del_number_of_for_statement'])

        features['number_of_try_statement'] = features['add_number_of_try_statement'] + features['del_number_of_try_statement']
        features = features.drop(columns=['add_number_of_try_statement','del_number_of_try_statement'])

        features['number_of_return_statement'] = features['add_number_of_return_statement'] + features['del_number_of_return_statement']
        features = features.drop(columns=['add_number_of_return_statement','del_number_of_return_statement'])

        features['number_of_single_variable_delcaration'] = features['add_number_of_single_variable_delcaration'] + features['del_number_of_single_variable_delcaration']
        features = features.drop(columns=['add_number_of_single_variable_delcaration','del_number_of_single_variable_delcaration'])

        features['number_of_qualified_name'] = features['add_number_of_qualified_name'] + features['del_number_of_qualified_name']
        features = features.drop(columns=['add_number_of_qualified_name','del_number_of_qualified_name'])

        features['for_flag'] = features['add_for_flag'] + features['del_for_flag']
        features = features.drop(columns=['add_for_flag','del_for_flag'])

        features['while_flag'] = features['add_while_flag'] + features['del_while_flag']
        features = features.drop(columns=['add_while_flag','del_while_flag'])

        features['do_flag'] = features['add_do_flag'] + features['del_do_flag']
        features = features.drop(columns=['add_do_flag','del_do_flag'])

        features['switch_flag'] = features['add_switch_flag'] + features['del_switch_flag']
        features = features.drop(columns=['add_switch_flag','del_switch_flag'])

        features['if_flag'] = features['add_if_flag'] + features['del_if_flag']
        features = features.drop(columns=['add_if_flag','del_if_flag'])

        features['try_flag'] = features['add_try_flag'] + features['del_try_flag']
        features = features.drop(columns=['add_try_flag','del_try_flag'])
    
    features.to_csv(features_path,index = False,header=True)
    
    print("Feature Creation for : " + type_of_change + " Done!")

if __name__ == "__main__":   
    
    type_of_changes = ['only_additions','only_deletions','both']
    data_split(data_folder_path + "removed_duplicates_files/",type_of_changes)
    
    for toc in type_of_changes:
        path = data_folder_path + 'data_split/' + toc
        merge_changes(path, toc)
    
    figures_directories_creation(type_of_changes)
    
    for toc in type_of_changes:
        features_calculation(toc)
    
    
   
