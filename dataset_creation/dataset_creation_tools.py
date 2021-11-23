import os
import sys
import traceback
import pandas as pd
from dataset_creation.astextractor import ASTExtractor
from pydriller import RepositoryMining
from pydriller.domain.commit import ModificationType
from properties import data_folder_path, ASTExtractorPath, ASTExtractorPropertiesPathDetailed, ASTExtractorPropertiesPath
from dataset_creation.xmlhelpers import get_lowest_common_ancestor, ast_to_xml_tree, get_node_ancestors, \
        find_difference_inserted, find_difference_inserted_deleted, code_to_ast, xml_to_str, close_xml_tags,\
        get_lowest_common_ancestor_additions_deletions
from itertools import islice
import collections,datetime
from difflib import SequenceMatcher
import xml.etree.ElementTree as ET

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_number_of_modified_lines(thecode):
    addeletions = [1 if line.startswith(
        "+") or line.startswith("-") else 0 for line in thecode.splitlines()[1:]]
    return sum(addeletions)
  
def get_difference_between_trees(tree1, tree2, has_inserts, has_deletes, return_also_place = False):
    inserts, deletes = [], []
    if has_inserts and has_deletes:
        for a, b in find_difference_inserted_deleted(tree1, tree2):
            if a == "insert": inserts.append(b)
            elif a == "delete": deletes.append(b)
        return inserts, deletes
    elif has_inserts:
        inserts = list(find_difference_inserted(tree1, tree2, return_also_place))
        if return_also_place:
            deletes = [i for i, _ in inserts]
            inserts = [i for _, i in inserts]
        return inserts, deletes
    elif has_deletes:
        deletes = list(find_difference_inserted(tree2, tree1, return_also_place))
        if return_also_place:
            inserts = [i for i, _ in deletes]
            deletes = [i for _, i in deletes]
        return inserts, deletes
    else:
        print("Error calling function get_difference_between_trees.")
        print("At least one of has_inserts and has_deletes must be True.")
        #print("Exiting...\n")
        return None,None
        #sys.exit(traceback.format_exc())

def get_difference_between_asts(ast1, ast2, has_additions, has_deletions, flag = False):
    
    tree1 = ast_to_xml_tree(ast1)
    tree2 = ast_to_xml_tree(ast2)
    inserts, deletes = get_difference_between_trees(tree1, tree2, has_additions, has_deletions)
    if flag == True:
        return xml_to_str(inserts,flag), xml_to_str(deletes,flag)
    else:
        if has_additions and has_deletions:
            inserts = get_lowest_common_ancestor(tree2, inserts, True)
            deletes = get_lowest_common_ancestor(tree1, deletes, True)
        elif has_additions:
            inserts = get_lowest_common_ancestor(tree2, inserts, True)
        else:
            deletes = get_lowest_common_ancestor(tree1, deletes, True)
        return xml_to_str(inserts,flag), xml_to_str(deletes,flag)

def get_additions_deletions_asts(ast1, ast2, has_additions, has_deletions,flag = False):
    tree1 = ast_to_xml_tree(ast1)
    tree2 = ast_to_xml_tree(ast2)
    inserts, deletes = get_difference_between_trees(tree1, tree2, has_additions, has_deletions)
    if flag == True:
        return xml_to_str(inserts,flag), xml_to_str(deletes,flag)
    else:
        if has_additions and has_deletions:
            inserts = get_lowest_common_ancestor_additions_deletions(tree2, inserts, True)
            deletes = get_lowest_common_ancestor_additions_deletions(tree1, deletes, True)
        elif has_additions:
            inserts = get_lowest_common_ancestor_additions_deletions(tree2, inserts, True)
        else:
            deletes = get_lowest_common_ancestor_additions_deletions(tree1, deletes, True)
        return xml_to_str(inserts,flag), xml_to_str(deletes,flag)

def get_method(tree1, tree1nodes):
    node_ancestors = get_node_ancestors(tree1, get_lowest_common_ancestor(tree1, tree1nodes))
    if node_ancestors != None:
        for ancestor in node_ancestors:              # If there are nested methods, node_ancestors makes sure
            if ancestor.tag == "MethodDeclaration":  # we get the outer one. Getting the inner one requires
                return ancestor                      # changing node_ancestors to reversed(node_ancestors)

def get_methods(node):
    def _find_rec(node, element):
        if node.tag == element:
            yield node
        else:
            for el in node.getchildren():
                yield from _find_rec(el, element)
    return _find_rec(node, "MethodDeclaration")

def get_methods_between_asts(code1, code2, ast1, ast2, has_additions, has_deletions,ast_extractor2):
    tree1 = ast_to_xml_tree(ast1)
    tree2 = ast_to_xml_tree(ast2)
    inserts, deletes = get_difference_between_trees(tree1, tree2, has_additions, has_deletions, True)

    method1 = get_method(tree1, deletes)
    method2 = get_method(tree2, inserts)

    if method1 != None and method2 != None:
        method1_index = [method for method in get_methods(tree1)].index(method1)
        method2_index = [method for method in get_methods(tree2)].index(method2)

        code1_ast = code_to_ast(ast_extractor2, code1)
        code1_tree = ast_to_xml_tree(code1_ast)
        methods1 = [method for method in get_methods(code1_tree)]

        code2_ast = code_to_ast(ast_extractor2, code2)
        code2_tree = ast_to_xml_tree(code2_ast)
        methods2 = [method for method in get_methods(code2_tree)]

        if len(methods1) == len(methods2) and method1_index >=0 and method1_index < len(methods1) and method2_index >=0 and method2_index < len(methods2):
            method1_code = methods1[method1_index]
            method2_code = methods2[method2_index]
            return method1_code.text, method2_code.text, xml_to_str(method1), xml_to_str(method2)
        else:
            return None, None, None, None
    else:
        return None, None, None, None

def split_code_patch_single_change(codepatch):
    try:
        code_before, code_after, code_additions, code_deletions = [], [], [], []
        for line in codepatch.splitlines():
            line = line.rstrip()
            if len(line.split("@@")) > 2:
                code_before.append(line.split("@@")[2][1:])
                code_after.append(line.split("@@")[2][1:])
            else:
                if line.startswith("+"):
                    code_additions.append(line[1:])
                    code_after.append(line[1:])
                elif line.startswith("-"):
                    code_deletions.append(line[1:])
                    code_before.append(line[1:])
                else:
                    code_before.append(line[1:])
                    code_after.append(line[1:])
        return "\n".join(code_before), "\n".join(code_after), "\n".join(code_additions), "\n".join(code_deletions)
    except:
        print("Could not split the following code patch:\n")
        print(codepatch + "\n")
        print("Exiting...\n")
        sys.exit(traceback.format_exc())

def split_code_patch_multiple_changes_into_dict(codepatch):
    """
    When code_diff contains multiple changes the variables are dictionaries containing each change
    individually.
    
    :param: codepatch: code_diff column drawn from pydriller library.
    
    :return:  code_before, code_after, code_additions, code_deletions, final_codepatch
    """
    
    try:
        # Creating the dictionaries in which we save the different changes.
        code_before_dict, code_after_dict, code_additions_dict, code_deletions_dict, final_codepatch_dict, code_only_dict, num_lines_modified_dict, num_lines_add_dict, num_lines_del_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}
        # Discarding the space that .split creates because the file always starts with @@.
        new_codepatch = codepatch.split('@@')[1:]
        for c, element in enumerate(new_codepatch):
            code_before, code_after, code_additions, code_deletions, code_only, num_lines_add, num_lines_del = [], [], [], [], [], [], []
            # This elements contain the number of lines modified, only saving them in final_codepatch for code_diff
            # data column.
            if c % 2 == 0:
                final_codepatch_dict[c] = "@@" + element + "@@"
                num_lines_modified_dict[c] = element
                try:
                    a = int(element.split(',')[0][2:]) 
                except:
                    # In some examples there are things like @@global.binlog_checksum in comments that create
                    # problems so we skip that files.
                    return None, None, None, None, None, None, None, None, None   
            # In this case element contains the code_diff.
            else:
                for k,line in enumerate(element.splitlines()):
                    if line.startswith("+"):
                        code_additions.append(line[1:])
                        num_lines_add.append(k)
                        code_after.append(line[1:])
                        code_only.append(line[1:])
                    elif line.startswith("-"):
                        code_deletions.append(line[1:])
                        num_lines_del.append(k)
                        code_before.append(line[1:])
                        code_only.append(line[1:])
                    else:
                        code_before.append(line[1:])
                        code_after.append(line[1:])
                        code_only.append(line[1:])
                # Adding @@...@@ with the code_diff for final_codepatch variables.
                final_codepatch_dict[c -1] = final_codepatch_dict[c-1] + element
                code_before_dict[c] = "\n".join(code_before)
                code_after_dict[c] = "\n".join(code_after)
                code_only_dict[c] = "\n".join(code_only)
                code_additions_dict[c] = "\n".join(code_additions)
                code_deletions_dict[c] = "\n".join(code_deletions)
                num_lines_add_dict[c] = num_lines_add
                num_lines_del_dict[c] = num_lines_del
        return code_before_dict, code_after_dict, code_additions_dict, code_deletions_dict, final_codepatch_dict,\
            code_only_dict,num_lines_modified_dict, num_lines_add_dict, num_lines_del_dict
    except:
        print("Could not split the following code patch:\n")
        print(codepatch + "\n")
        print("Exiting...\n")
        sys.exit(traceback.format_exc())

def code_after_for_single_change_after_removing_duplicates(code_before,dict_list):
    
    code_after =[]
    counter = 0
    addition_counter = 0
    deletion_counter = 0
    lines = iter(enumerate(code_before.splitlines()))
    for c1, line in lines:
        if c1 >= (int(dict_list[6][0].split(',')[0][2:]) - 2):
            # This condition mean that all additions and deletions have been updated to codes_split_after
            # so we keep copying the lines remaining till the end of the file.
            if addition_counter == len(dict_list[7][1]) and deletion_counter == len(dict_list[8][1]):
                code_after.append(line)
            else:
                # Counter is placed here becuase the indexes of additions and deletions in
                # variable split starts from 1 and not from 0.
                counter = counter + 1
                # flag0 tell us if the work is done inside the two ifs below or it should
                # be done by the third one.
                flag0 = True
                # Counter for multi deletions.
                multi_del = 0
                multi_add = 0

                # flag1 tells us if there has been a deletion done allready
                flag1 = False
                # split[8] contains deletions so here we check if there are any deletions.
                if len(dict_list[8][1]) > 0:
                    for c_mult, i in enumerate(dict_list[8][1]):
                        if counter == i:
                            if len(dict_list[8][1]) == 1:
                                # In this case we are sure that line appends first and then the addition.
                                code_after.append(line)
                                deletion_counter = deletion_counter + 1
                                counter = counter + 1
                                flag1 = True
                                consume(lines, 1)
                            # Otherwise we need the c2 addition or c2 + multi_add additions in a row.
                            else:
                                code_after.append(line)
                                for j in range(len(dict_list[8][1][c_mult + 1:])):
                                    if c_mult + (j+1) <= len(dict_list[8][1]):
                                        if dict_list[8][1][c_mult+j] == dict_list[8][1][c_mult+j+1] - 1:
                                            multi_del = multi_del + 1
                                        else:
                                            break

                                for j in range(multi_del + 1):
                                    deletion_counter = deletion_counter + 1
                                    counter = counter + 1
                                flag1 = True
                                consume(lines, multi_del + 1)
                            flag0 = False

                # split[7] contains additions so here we check if there are any additions.
                if len(dict_list[7][1]) > 0:
                    for c2, i in enumerate(dict_list[7][1]):
                        if counter == i:

                            # split[7] contains additions, if the addition is only 1 we just add it after line.
                            if len(dict_list[7][1]) == 1:
                                # In this case we are sure that line appends first and then the addition.
                                if not flag1:
                                    code_after.append(line)
                                code_after.append(dict_list[2][1])
                                addition_counter = addition_counter + 1
                                counter = counter + 1
                            # Otherwise we need the c2 addition or c2 + multi_add additions in a row.
                            else:
                                for j in range(len(dict_list[7][1][c2 + 1:])):
                                    if c2 + (j+1) <= len(dict_list[7][1]):
                                        if dict_list[7][1][c2+j] == dict_list[7][1][c2+j+1] - 1:
                                            multi_add = multi_add + 1
                                        else:
                                            break
                                temp = dict_list[2][1].split('\n')
                                # This check tells us if a deletion has allready been made because
                                # then we dont need to append(line) again we just skip it
                                if not flag1:
                                    code_after.append(line)
                                for j in range(multi_add + 1):
                                    code_after.append(temp[c2+j])
                                    addition_counter = addition_counter + 1
                                    counter = counter + 1
                            flag0 = False
                if flag0 == True:
                    code_after.append(line)
        else:
            code_after.append(line)
    
    
    
    return "\n".join(code_after)

def split_code_diffs(code_before, dict_list):
    """
    This function splits multiple code_diffs of a modification in individual modifications.

    :param: code_before: containing the whole code of the file before the modification.
    :param: dict_list: containing all variables calculated by split_code_patch_multiple_changes_into_dict

    :return: final_codes_split_after: containing code_after for every individual code_diff after the split.
    :return: final_codes_split_diff: containing the individual code_diffs.
    """                   
    # Creating the indexes for split diffs.
    split_indexes = []
    for k in dict_list[0]:
        split_indexes.append(k)
         
    temp_split_dicts_list = []
    temp_split_dict0 = {}
    temp_split_dict1 = {}
    temp_split_dict2 = {}
    temp_split_dict3 = {}
    temp_split_dict4 = {}
    temp_split_dict5 = {}
    temp_split_dict6 = {}
    temp_split_dict7 = {}
    temp_split_dict8 = {}
    for c, i in enumerate(split_indexes):
        temp_split_dict0[c] = dict_list[0][i]
        temp_split_dict1[c] = dict_list[1][i]
        temp_split_dict2[c] = dict_list[2][i]
        temp_split_dict3[c] = dict_list[3][i]
        temp_split_dict4[c] = dict_list[4][i-1]
        temp_split_dict5[c] = dict_list[5][i]
        temp_split_dict6[c] = dict_list[6][i-1]
        temp_split_dict7[c] = dict_list[7][i]
        temp_split_dict8[c] = dict_list[8][i]
        temp_split_dicts_list.append([temp_split_dict0[c], temp_split_dict1[c], temp_split_dict2[c], temp_split_dict3[c], temp_split_dict4[c],
                                    temp_split_dict5[c], temp_split_dict6[c], temp_split_dict7[c], temp_split_dict8[c]])

    codes_split_after = [[] for _ in range(len(temp_split_dicts_list))]
    final_codes_split_after = []
    final_codes_split_diff = []
    for c0, split in enumerate(temp_split_dicts_list):
        lines_modified = split[6]
        counter = 0
        addition_counter = 0
        deletion_counter = 0            
        lines = iter(enumerate(code_before.splitlines()))
        for c1, line in lines:
            # I think that from pydriller, code_diff begins 2 lines above the numbers written in @@....@@, that's why
            # I subtract number 2 from lines_modified below.
            if c1 >= (int(lines_modified.split(',')[0][2:]) - 2):
                # This condition mean that all additions and deletions have been updated to codes_split_after
                # so we keep copying the lines remaining till the end of the file.
                if addition_counter == len(split[7]) and deletion_counter == len(split[8]):
                    codes_split_after[c0].append(line)
                else:
                    # Counter is placed here becuase the indexes of additions and deletions in
                    # variable split starts from 1 and not from 0.
                    counter = counter + 1
                    # flag0 tell us if the work is done inside the two ifs below or it should
                    # be done by the third one.
                    flag0 = True
                    # Counter for multi deletions.
                    multi_del = 0
                    multi_add = 0
                    
                    # flag1 tells us if there has been a deletion done allready
                    flag1 = False
                    # split[8] contains deletions so here we check if there are any deletions.
                    if len(split[8]) > 0:
                        for c_mult,i in enumerate(split[8]):
                            if counter == i:
                                if len(split[8]) == 1:
                                    # In this case we are sure that line appends first and then the addition.
                                    codes_split_after[c0].append(line)
                                    deletion_counter = deletion_counter + 1
                                    counter = counter + 1
                                    flag1 = True
                                    consume(lines,1)
                                # Otherwise we need the c2 addition or c2 + multi_add additions in a row.
                                else:
                                    codes_split_after[c0].append(line)
                                    for j in range(len(split[8][c_mult + 1 :])):
                                        if c_mult + (j+1) <= len(split[8]):
                                            if split[8][c_mult+j] == split[8][c_mult+j+1] - 1:
                                                multi_del = multi_del + 1
                                            else:
                                                break
                                    
                                    for j in range(multi_del + 1):
                                        deletion_counter = deletion_counter + 1
                                        counter = counter + 1
                                    flag1 = True
                                    consume(lines, multi_del + 1)
                                    
                                flag0 = False
                                
                    # split[7] contains additions so here we check if there are any additions.
                    if len(split[7]) > 0:
                        for c2, i in enumerate(split[7]):
                            if counter == i:
                                    
                                # split[7] contains additions, if the addition is only 1 we just add it after line.
                                if len(split[7]) == 1:
                                    # In this case we are sure that line appends first and then the addition.
                                    if not flag1:
                                        codes_split_after[c0].append(line)
                                    codes_split_after[c0].append(split[2])
                                    addition_counter = addition_counter + 1
                                    counter = counter + 1
                                # Otherwise we need the c2 addition or c2 + multi_add additions in a row.
                                else:
                                    for j in range(len(split[7][c2 + 1 :])):
                                            if c2 + (j+1) <= len(split[7]):
                                                if split[7][c2+j] == split[7][c2+j+1] - 1:
                                                    multi_add = multi_add + 1
                                                else:
                                                    break
                                    temp = split[2].split('\n')
                                    # This check tells us if a deletion has allready been made because
                                    # then we dont need to append(line) again we just skip it
                                    if not flag1:
                                        codes_split_after[c0].append(line)
                                    for j in range(multi_add + 1):
                                        codes_split_after[c0].append(temp[c2+j])
                                        addition_counter = addition_counter + 1
                                        counter = counter + 1
                                flag0 = False
                    if flag0 == True:
                        codes_split_after[c0].append(line)
            else:
                codes_split_after[c0].append(line)

        final_codes_split_diff.append(split[4])
    
    for code in codes_split_after:
        final_codes_split_after.append("\n".join(code))
        #print(final_codes_split_after)
    
    return final_codes_split_after,final_codes_split_diff

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def get_final_asts(ast_extractor,ast_extractor2,code_before,code_after,code_diff,success_counter,failed_counter):
    code_before_ast = code_to_ast(ast_extractor, code_before)
    code_after_ast = code_to_ast(ast_extractor, code_after)
    if code_before_ast == '' or code_after_ast == '':
        return '','',None,None,None,None,None,None,None,None,success_counter,failed_counter
    _, _, code_additions, code_deletions = split_code_patch_single_change(code_diff)
    code_additions_ast, code_deletions_ast, inserts_AST, deletes_AST = '-1', '-1','-1','-1'

    has_additions = any(line.lstrip().startswith("+") for line in code_diff.splitlines())
    has_deletions = any(line.lstrip().startswith("-") for line in code_diff.splitlines())
    
    method_code_before, method_code_after, method_code_before_ast, method_code_after_ast = \
                        get_methods_between_asts(code_before, code_after, code_before_ast, code_after_ast, has_additions, has_deletions, ast_extractor2)
    if method_code_before != None:
        code_additions_ast, code_deletions_ast = get_additions_deletions_asts(method_code_before_ast, method_code_after_ast, has_additions, has_deletions)
        code_inserts_ast, code_deletes_ast = get_additions_deletions_asts(method_code_before_ast, method_code_after_ast, has_additions, has_deletions,True)
        code_inserts_ast = close_xml_tags(code_inserts_ast)
        code_deletes_ast = close_xml_tags(code_deletes_ast)
        code_additions_ast = close_xml_tags(code_additions_ast)
        code_deletions_ast = close_xml_tags(code_deletions_ast)
        success_counter = success_counter + 1
        return method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                    code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,success_counter,failed_counter
    else:
        failed_counter = failed_counter + 1
        return None,None,None,None,None,None,None,None,None,None,success_counter,failed_counter

def save_data(method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,sha,filename,\
                            repo_name,date,message,code_diff,start,commits_path,success_check_counter,failed_check_counter):
    check = lambda x, y: x != -1 and x != "-1" and x != None and y != -1 and y != "-1" and y != None
    if (check(method_code_before, method_code_before_ast) and check(method_code_after, method_code_after_ast)) and \
                    (check(code_additions, code_additions_ast) or check(code_deletions, code_deletions_ast)): 
        success_check_counter = success_check_counter + 1
        if code_additions == '' or "" :
            code_additions = '-1'
        if code_deletions == '' or "" :
            code_deletions = '-1'
        if code_additions == -1 :
            code_additions = '-1'
        if code_deletions == -1 :
            code_deletions = '-1'
        if code_additions_ast == -1:
            code_additions_ast = '-1'
        if code_deletions_ast == -1:
            code_deletions_ast = '-1'
        res = ['\\x'+sha, filename, repo_name, date, message, code_diff, method_code_before, method_code_after, method_code_before_ast, method_code_after_ast, \
                    code_additions, code_deletions, code_additions_ast, code_deletions_ast, code_inserts_ast, code_deletes_ast]
        df = pd.DataFrame.from_records([res], columns=["sha", "filename", "repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                    "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])
        df = df.set_index("sha", drop=True)
        if start == False:
            df.to_csv(commits_path, sep='\t',encoding='utf-8')
            start = True
        else:
            df.to_csv(commits_path, sep='\t', encoding='utf-8', header=None, mode="a")
    else:
        failed_check_counter = failed_check_counter + 1
    return start ,success_check_counter, failed_check_counter
