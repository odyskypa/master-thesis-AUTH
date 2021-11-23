import os
import sys
import traceback
import pandas as pd
from dataset_creation.astextractor import ASTExtractor
from pydriller import RepositoryMining
from pydriller.domain.commit import ModificationType
from properties import data_folder_path, ASTExtractorPath, ASTExtractorPropertiesPathDetailed, ASTExtractorPropertiesPath
from dataset_creation.dataset_creation_tools import *
from tree_distance import parallel

def extract_commits_info(repo_sha):
    ast_extractor = ASTExtractor(ASTExtractorPath, ASTExtractorPropertiesPathDetailed)
    ast_extractor2 = ASTExtractor(ASTExtractorPath, ASTExtractorPropertiesPath)
    
    commits_total_counter = 0
    modifications_total_counter = 0
    changes_total_counter = 0
    changes_removed_similars = 0
    modifications_type_skip_counter = 0
    modifications_length_skip_counter = 0
    modifications_single_change_counter = 0
    modifications_multi_changes_counter = 0
    single_modifications_success_get_method_counter = 0
    single_modifications_failed_get_method_counter = 0
    single_modifications_empty_method_skip_counter = 0
    single_modifications_success_check_counter = 0
    single_modifications_failed_check_counter = 0
    multi_modifications_papakia_skip_counter = 0
    single_modifications_success_get_method_after_similar_removed = 0
    single_modifications_failed_get_method_after_similar_removed = 0
    single_modifications_empty_method_after_similar_removed = 0
    single_modifications_success_check_after_similar_removed = 0
    single_modifications_failed_check_after_similar_removed = 0
    multi_modifications_success_get_method_split_counter = 0
    multi_modifications_failed_get_method_split_counter = 0
    multi_modifications_empty_method_skip_counter = 0
    multi_modifications_success_check_counter = 0
    multi_modifications_failed_check_counter = 0
    
    
    repo_name = repo_sha[:-9]
    print("Extracting commits info for project: " + repo_name)
    git_repo_path = os.path.join(data_folder_path, repo_name)
    sha_path = os.path.join(data_folder_path + "shas_files/", repo_sha)
    commits_path = os.path.join(data_folder_path + "commits_files/", repo_name + "_commits.csv")
    # Na to diagrapsw meta ayto to if
    if os.path.exists(commits_path):
        repo_counters = {'commits_total_counter':commits_total_counter,\
                    'modifications_total_counter':modifications_total_counter,\
                    'modifications_total_skip_counter': 0,\
                    'changes_total_counter': changes_total_counter,\
                    'modifications_type_skip_counter':modifications_type_skip_counter,\
                    'modifications_length_skip_counter':modifications_length_skip_counter,\
                    'changes_removed_similars':changes_removed_similars,\
                    'modifications_single_change_counter':modifications_single_change_counter,\
                    'modifications_multi_changes_counter':modifications_multi_changes_counter,\
                    'single_modifications_success_get_method_counter':single_modifications_success_get_method_counter,\
                    'single_modifications_failed_get_method_counter':single_modifications_failed_get_method_counter,\
                    'single_modifications_empty_method_skip_counter':single_modifications_empty_method_skip_counter,\
                    'single_modifications_success_check_counter':single_modifications_success_check_counter,\
                    'single_modifications_failed_check_counter':single_modifications_failed_check_counter,\
                    'multi_modifications_papakia_skip_counter':multi_modifications_papakia_skip_counter,\
                    'single_modifications_success_get_method_after_similar_removed':single_modifications_success_get_method_after_similar_removed,\
                    'single_modifications_failed_get_method_after_similar_removed':single_modifications_failed_get_method_after_similar_removed,\
                    'single_modifications_empty_method_after_similar_removed':single_modifications_empty_method_after_similar_removed,\
                    'single_modifications_success_check_after_similar_removed':single_modifications_success_check_after_similar_removed,\
                    'single_modifications_failed_check_after_similar_removed':single_modifications_failed_check_after_similar_removed,\
                    'multi_modifications_success_get_method_split_counter':multi_modifications_success_get_method_split_counter,\
                    'multi_modifications_failed_get_method_split_counter':multi_modifications_failed_get_method_split_counter,\
                    'multi_modifications_empty_method_skip_counter':multi_modifications_empty_method_skip_counter,\
                    'multi_modifications_success_check_counter':multi_modifications_success_check_counter,\
                    'multi_modifications_failed_check_counter':multi_modifications_failed_check_counter}
        df = pd.DataFrame(repo_counters, index=[repo_name])
        return df

    with open(sha_path) as infile:
        shas = [line.strip() for line in infile if line]
    repomining = RepositoryMining(git_repo_path, only_commits=shas)

    print("Processing " + str(len(shas)) + " commits...")
    start = False
    commits = iter(enumerate(repomining.traverse_commits()))
    
    for c, commit in commits:
        commits_total_counter = commits_total_counter + 1
        modifications_total_counter = modifications_total_counter + len(commit.modifications)
        for k,modification in enumerate(commit.modifications):
            if modification.change_type == ModificationType.MODIFY and modification.filename.endswith('.java'):
                code_diff = '\n'.join(modification.diff.splitlines())
                if get_number_of_modified_lines(code_diff)<= 100:
                    sha = commit.hash
                    filename = modification.new_path
                    date = commit.committer_date
                    message = commit.msg
                    code_diff = "\n".join(modification.diff.splitlines()).encode('ascii', 'ignore').decode()
                    code_before = "\n".join(modification.source_code_before.splitlines()).encode('ascii', 'ignore').decode()
                    
                    if len(code_diff.split("@@")) < 4:
                        modifications_single_change_counter = modifications_single_change_counter + 1
                        changes_total_counter = changes_total_counter + 1
                        code_after = "\n".join(modification.source_code.splitlines()).encode('ascii', 'ignore').decode()
                        try:
                            method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                                    code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,\
                                        single_modifications_success_get_method_counter,\
                                            single_modifications_failed_get_method_counter\
                                            = get_final_asts(ast_extractor,ast_extractor2,code_before,code_after,code_diff,\
                                                    single_modifications_success_get_method_counter,single_modifications_failed_get_method_counter)
                            if method_code_before == '':
                                single_modifications_empty_method_skip_counter = single_modifications_empty_method_skip_counter + 1
                                continue   
                        except Exception as e:
                            print("Could not parse commit " + sha + " of repo " + repo_name + " with the following patch:\n")
                            print(code_diff + "\n")
                            print("\n")
                            print(e)
                            print("\n")
                            continue
                            #sys.exit(traceback.format_exc())

                        start,single_modifications_success_check_counter,single_modifications_failed_check_counter\
                            = save_data(method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                                        code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,\
                                            sha,filename,repo_name,date,message,code_diff,start,commits_path,single_modifications_success_check_counter,\
                                                        single_modifications_failed_check_counter)
                    else:
                        modifications_multi_changes_counter = modifications_multi_changes_counter + 1
                        code_before_dict, code_after_dict, code_additions_dict, code_deletions_dict, code_diff_dict,\
                                        code_only_dict,num_lines_modified_dict, num_lines_add_dict, num_lines_del_dict \
                                                = split_code_patch_multiple_changes_into_dict(code_diff)
                        
                        # In this case something with split with @@ has gone wrong because
                        # there might exist a comment with @@ inside or something like this
                        # so we skip this modification.
                        # All 9 variables above are None but we just check the first one.
                        if code_before_dict == None :
                            multi_modifications_papakia_skip_counter = multi_modifications_papakia_skip_counter + 1
                            continue
                        dict_list = [code_before_dict, code_after_dict, code_additions_dict, code_deletions_dict, code_diff_dict,\
                                code_only_dict,num_lines_modified_dict, num_lines_add_dict, num_lines_del_dict]
                        
                        changes_total_counter = changes_total_counter + len(dict_list[0])
                        
                        scalar_indexes = list(range(len(dict_list[0])))
                        if len(dict_list[0]) > 0 :
                            sim_indexes = []
                            for c,(key,add) in enumerate(dict_list[2].items()):
                                for c1,(key1,add1) in enumerate(dict_list[2].items()):
                                    # Skipping keys that allready have been checked in loop.
                                    if key >= key1:
                                        continue
                                    deli = dict_list[3][key]
                                    deli1 = dict_list[3][key1]
                                    """ print(similar(add, add1))
                                    print(similar(deli,deli1))
                                    print(similar(add1, add))
                                    print(similar(deli1,deli)) """
                                    if (( similar(add, add1) >= 0.80 and similar(deli,deli1) >= 0.80 ) or (similar(add1, add) >= 0.80 and similar(deli1,deli) >= 0.80)):
                                        sim_indexes.append((c,c1))
                        
                            if len(sim_indexes) > 0:
                                for i in range(len(sim_indexes)):
                                    for j in range(len(sim_indexes[i])):
                                        if j == max(range(len(sim_indexes[i]))):
                                            continue
                                        if sim_indexes[i][j+1] in scalar_indexes:
                                            scalar_indexes.remove(sim_indexes[i][j+1])
                                            changes_removed_similars = changes_removed_similars + 1

                                if len(scalar_indexes) != len(dict_list[0]):
                                    for dictionary in dict_list:
                                        for i , key in enumerate(dictionary.copy()):
                                            if i not in scalar_indexes:
                                                dictionary.pop(key)
                        # In this case after removing duplicates there is only one change in the file so we save it as we did
                        # above for single changes after the calculation of new code_after_final.
                        if len(dict_list[4]) == 1 :
                            code_after = code_after_for_single_change_after_removing_duplicates(code_before,dict_list)
                            code_diff = dict_list[4][0]
                            try:
                                method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                                    code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,\
                                        single_modifications_success_get_method_after_similar_removed,\
                                            single_modifications_failed_get_method_after_similar_removed = get_final_asts(ast_extractor,ast_extractor2,code_before,code_after,code_diff,\
                                                                    single_modifications_success_get_method_after_similar_removed,single_modifications_failed_get_method_after_similar_removed)
                                if method_code_before == '':
                                    single_modifications_empty_method_after_similar_removed = single_modifications_empty_method_after_similar_removed + 1
                                    continue    
                            except Exception as e:
                                print("Could not parse commit " + sha + " of repo " + repo_name + " with the following patch:\n")
                                print(code_diff + "\n")
                                print("\n")
                                print(e)
                                print("\n")
                                continue
                                #sys.exit(traceback.format_exc())
                                
                            start,single_modifications_success_check_after_similar_removed,single_modifications_failed_check_after_similar_removed\
                                = save_data(method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                                        code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,\
                                            sha,filename,repo_name,date,message,code_diff,start,commits_path,single_modifications_success_check_after_similar_removed,\
                                                        single_modifications_failed_check_after_similar_removed)
                        else:
                            final_codes_split_after, final_codes_split_diff = split_code_diffs(code_before, dict_list)
                            for c,code_after in enumerate(final_codes_split_after):
                                code_diff = final_codes_split_diff[c]
                                try:
                                    method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                                    code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,\
                                        multi_modifications_success_get_method_split_counter,\
                                            multi_modifications_failed_get_method_split_counter = get_final_asts(ast_extractor,ast_extractor2,code_before,code_after,code_diff,\
                                                                                            multi_modifications_success_get_method_split_counter,multi_modifications_failed_get_method_split_counter)
                                    if method_code_before == '':
                                        multi_modifications_empty_method_skip_counter = multi_modifications_empty_method_skip_counter + 1
                                        continue
                                except Exception as e:
                                    print("Could not parse commit " + sha + " of repo " + repo_name + " with the following patch:\n")
                                    print(code_diff + "\n")
                                    print("\n")
                                    print(e)
                                    print("\n")
                                    continue
                                    #sys.exit(traceback.format_exc())
                                
                                start,multi_modifications_success_check_counter,multi_modifications_failed_check_counter\
                                    = save_data(method_code_before,method_code_before_ast,method_code_after,method_code_after_ast,\
                                            code_additions,code_additions_ast,code_deletions,code_deletions_ast,code_inserts_ast,code_deletes_ast,sha,\
                                                filename,repo_name,date,message,code_diff,start,commits_path,multi_modifications_success_check_counter,\
                                                    multi_modifications_failed_check_counter)
                else:
                    modifications_length_skip_counter = modifications_length_skip_counter + 1
            else:
                modifications_type_skip_counter = modifications_type_skip_counter + 1
    
    modifications_total_skip_counter =  modifications_type_skip_counter + modifications_length_skip_counter + single_modifications_empty_method_skip_counter\
                                        + multi_modifications_papakia_skip_counter + single_modifications_empty_method_after_similar_removed\
                                            + multi_modifications_empty_method_skip_counter
    repo_counters = {'commits_total_counter':commits_total_counter,\
                    'modifications_total_counter':modifications_total_counter,\
                    'modifications_total_skip_counter': modifications_total_skip_counter,\
                    'changes_total_counter': changes_total_counter,\
                    'modifications_type_skip_counter':modifications_type_skip_counter,\
                    'modifications_length_skip_counter':modifications_length_skip_counter,\
                    'changes_removed_similars':changes_removed_similars,\
                    'modifications_single_change_counter':modifications_single_change_counter,\
                    'modifications_multi_changes_counter':modifications_multi_changes_counter,\
                    'single_modifications_success_get_method_counter':single_modifications_success_get_method_counter,\
                    'single_modifications_failed_get_method_counter':single_modifications_failed_get_method_counter,\
                    'single_modifications_empty_method_skip_counter':single_modifications_empty_method_skip_counter,\
                    'single_modifications_success_check_counter':single_modifications_success_check_counter,\
                    'single_modifications_failed_check_counter':single_modifications_failed_check_counter,\
                    'multi_modifications_papakia_skip_counter':multi_modifications_papakia_skip_counter,\
                    'single_modifications_success_get_method_after_similar_removed':single_modifications_success_get_method_after_similar_removed,\
                    'single_modifications_failed_get_method_after_similar_removed':single_modifications_failed_get_method_after_similar_removed,\
                    'single_modifications_empty_method_after_similar_removed':single_modifications_empty_method_after_similar_removed,\
                    'single_modifications_success_check_after_similar_removed':single_modifications_success_check_after_similar_removed,\
                    'single_modifications_failed_check_after_similar_removed':single_modifications_failed_check_after_similar_removed,\
                    'multi_modifications_success_get_method_split_counter':multi_modifications_success_get_method_split_counter,\
                    'multi_modifications_failed_get_method_split_counter':multi_modifications_failed_get_method_split_counter,\
                    'multi_modifications_empty_method_skip_counter':multi_modifications_empty_method_skip_counter,\
                    'multi_modifications_success_check_counter':multi_modifications_success_check_counter,\
                    'multi_modifications_failed_check_counter':multi_modifications_failed_check_counter}
    df = pd.DataFrame(repo_counters, index=[repo_name])
    print("Repo:" + repo_name +  "Done!")

    ast_extractor.close()
    ast_extractor2.close()
    return df

if __name__ == "__main__":
    
    begin_time = datetime.datetime.now()
    
    
    if not os.path.exists(data_folder_path + "commits_files/"):
        os.makedirs(data_folder_path + "commits_files/")
    repo_shas = [repo_sha for repo_sha in os.listdir(data_folder_path + "shas_files/") if repo_sha.endswith("shas.csv")]
    counters_path = os.path.join(data_folder_path + "s2_counters.csv")
    
    
    start0 = False
    for repo_sha in repo_shas:
        df = extract_commits_info(repo_sha)
        if start0 == False:
            df.to_csv(counters_path)
            start0 = True
        else:
            df.to_csv(counters_path, header=None, mode="a")
        
    print("Time for s2 to be completed: ",datetime.datetime.now() - begin_time)
