import os
import re
import csv
import math
import pandas as pd
from collections import Counter
from properties import data_test_folder_path,data_folder_path
from dataset_creation.dataset_creation_tools import similar

csv.field_size_limit(50000000)

def get_cosine(vec1,vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x]*vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator)/ denominator

def isNaN(num):
    var = False
    if num!= num:
        var = True
    elif num == '':
        var = True
    return var

def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)

def search(commit_files,log_path,toc,centroid_flag):
    if centroid_flag == True:
        dir = "centroids"
    else:
        dir = "clusters"
    
    for c1,data_test_file_name in enumerate(commit_files):
            log_outfile = open(log_path,'a')
            print('This is the ' + str(c1) + ' file of testing commit files.')    
            
            data_test_path = data_test_folder_path + "commits_files/"  + data_test_file_name
            data = pd.read_csv(data_test_path, delimiter='\t')
                            
            print('Length of test data is : ',len(data))
            
            for i,test_row in data.iterrows():

                print('Current test_row of data is : ', i)

                input_code_additions = test_row['code_additions']
                input_code_deletions = test_row['code_deletions']
                flag_only_add = False
                flag_only_del = False
                if isNaN(input_code_additions) and isNaN(input_code_deletions):
                    continue
                elif isNaN(input_code_additions):
                    input_code_additions = 'NaN'
                    flag_only_del = True
                elif isNaN(input_code_deletions):
                    input_code_deletions = 'NaN'
                    flag_only_add = True
                precluster_names = os.listdir(data_folder_path + 'clustering/' + dir + '/' + toc)
                for name in precluster_names:
                    for cluster_name in os.listdir(data_folder_path + 'clustering/' + dir + '/' + toc + '/' + name):
                        
                        cluster_data_path = data_folder_path + '/clustering/' + dir + '/' + toc + '/' + name + '/' + cluster_name
                        cluster_data = pd.read_csv(cluster_data_path,index_col=0)
                        
                        for i2,test_row2 in cluster_data.iterrows():
                            
                            cluster_code_additions = test_row2['code_additions']
                            cluster_code_deletions = test_row2['code_deletions']
                            addFlag = False
                            delFlag = False
                            if toc == "only_additions":
                                cluster_code_deletions ='NaN'
                                addFlag = True
                            if toc == "only_deletions":
                                cluster_code_additions = 'NaN'
                                delFlag = True
                            addition_similarity = similar(input_code_additions,cluster_code_additions)
                            deletion_similarity = similar(input_code_deletions,cluster_code_deletions)
                            if  (flag_only_add and addition_similarity >= 0.7) or\
                                        (flag_only_del and deletion_similarity >= 0.7) or\
                                        (addFlag and addition_similarity >=0.7) or\
                                        (delFlag and deletion_similarity >=0.7) or\
                                        (addition_similarity >= 0.7 and deletion_similarity >= 0.7):
                                print(' FOUND ONE !!! ')
                                log_outfile.write(data_test_file_name +'\n')
                                log_outfile.write(str(i) + '\n')
                                log_outfile.write(name + '\n')
                                log_outfile.write(cluster_name + '\n')
                                log_outfile.write(str(i2) + '\n')
            log_outfile.close()
                
def finding_test_file_in_clusters(centroid_flag,toc):
    log_name = toc + str(centroid_flag) + "_log.csv"
    log_path = os.path.join(data_test_folder_path,log_name)
      
    commit_files = [commit_file for commit_file in os.listdir(data_test_folder_path + "commits_files/") if commit_file.endswith("commits.csv")]  
    search(commit_files,log_path,toc,centroid_flag)
            
def final_results(centroid_flag,toc):
    
    
    final_path = data_test_folder_path + toc + str(centroid_flag) +'_final.csv'
    
    if centroid_flag == True:
        dir = "centroids"
    else:
        dir = "clusters"
    header = False
    output_file = open(final_path,'a',newline='', encoding='utf-8')
    writer = csv.writer(output_file)
    row=["data_test_file_name","test_index","method_code_diff_test", "method_code_before_cluster","method_code_after_cluster", "code_deletions_cluster",\
        "code_additions_cluster", "input_code_deletion_test", "input_code_addition_test",\
        "index_of_cluster", "name_of_cluster", "toc",\
        "name_of_precluster", "addition_similarity", "deletion_similarity"]
    writer.writerow(row)
    output_file.close()
    csv_file = open(data_test_folder_path + toc + str(centroid_flag) + '_log.csv','r')
    reader = csv.reader(csv_file)
    my_list = list(reader)
    csv_file.close()
    c1 = 0
    c2 = 5
    while c2 < len(my_list):
        
        output_file = open(final_path,'a',newline='', encoding='utf-8')
        writer = csv.writer(output_file)   
        
        lines = my_list[c1:c2]
        
        data_test_file_name = str(lines[0][0])
        test_index = int(lines[1][0])    
        same_example_precluster = lines[2][0]
        same_example_name = lines[3][0]
        same_example_index = int(lines[4][0])
        
        data_test_path = data_test_folder_path + "commits_files/"  + data_test_file_name
        
        
        try:
            data_file = open(data_test_path,'r')
            data_reader = csv.reader(data_file,dialect='excel', delimiter='\t')
            test_row = list(data_reader)[test_index + 1]
            data_file.close()
            
            cluster_data_path = data_folder_path + "/clustering/" + dir + '/' + toc + '/' + same_example_precluster +'/' + same_example_name
            cluster_data = pd.read_csv(cluster_data_path,index_col=0)
            

            same_example = cluster_data.loc[[same_example_index]]
            same_example_method_code_before = same_example['method_code_before'].values[0]
            same_example_method_code_after = same_example['method_code_after'].values[0]
            same_example_code_deletions = same_example['code_deletions'].values[0]
            same_example_code_additions = same_example['code_additions'].values[0]
            
            method_code_diff_test = test_row[5]
            input_code_deletion = test_row[9]
            input_code_addition = test_row[8]
            
            if toc == "only_additions":
                same_example_code_deletions ='NaN'
            if toc == "only_deletions":
                same_example_code_additions = 'NaN'
            
            if isNaN(input_code_addition) and isNaN(input_code_deletion):
                c1 = c1 + 5
                c2 = c2 + 5
                output_file.close()
            else:
                if isNaN(input_code_addition):
                    input_code_addition = 'NaN'
                elif isNaN(input_code_deletion):
                    input_code_deletion = 'NaN'
                
                addition_similarity = similar(input_code_addition,same_example_code_additions)
                deletion_similarity = similar(input_code_deletion,same_example_code_deletions)
                
                row = [data_test_file_name,test_index,method_code_diff_test,same_example_method_code_before,same_example_method_code_after,\
                        same_example_code_deletions,same_example_code_additions,input_code_deletion,input_code_addition,\
                        same_example_index, same_example_name,toc,same_example_precluster,addition_similarity,\
                        deletion_similarity]
                writer.writerow(row)
                output_file.close()
                print(c2," out of :",len(my_list))
                c1 = c1 + 5
                c2 = c2 + 5   
        except Exception as e:
            print(e)
            output_file.close()
            c1 = c1 + 5
            c2 = c2 + 5
    print("Finished")