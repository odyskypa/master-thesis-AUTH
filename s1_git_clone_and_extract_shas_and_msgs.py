import os
import subprocess
import pandas as pd
from pydriller.domain.commit import ModificationType
from pydriller import RepositoryMining, GitRepository
from properties import data_folder_path,git_executable_path
from tree_distance import parallel

def get_number_of_modified_lines(thecode):
	addeletions = [1 if line.startswith("+") or line.startswith("-") else 0 for line in thecode.splitlines()[1:]]
	return sum(addeletions)

def process_repo(repo_address):
    repo_name = '_'.join(repo_address.split('/')[-2:])
    print('\nProcessing repo ' + repo_name)
    #data_folder_path = 'C:/Users/Odyskypa/Desktop/data1/'
    # String_list containing words that are more usable by programmers when fixing problems as show in the latest paper of bibliography.
    fix_string_list =['fix','improve','change','bug','add','remove','support']
    
    #Set paths
    git_repo_path = os.path.join(data_folder_path, repo_name)
    sha_path = os.path.join(data_folder_path + "shas_files/", repo_name + "_shas.csv")
    msg_path = os.path.join(data_folder_path + "msgs_files/", repo_name + "_msgs.csv")
    
    if not os.path.exists(git_repo_path):
        print("Downloading....")
        #Download repo
        p = subprocess.Popen([git_executable_path, 'clone', repo_address,git_repo_path],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        while True:
            line = p.stdout.readline()
            if line == b'':
                break
        print(" Downloading Done !!!")
    
    #Get repo commits
    try:
        total = GitRepository(git_repo_path).total_commits()
        print('Processing ' + str(total) + ' commits....')
        repomining = RepositoryMining(git_repo_path, only_modifications_with_file_types=['.java'])
        with open(sha_path,'w') as sha_outfile, open(msg_path,'w') as msg_outfile :
            for commit in repomining.traverse_commits():
                if commit.in_main_branch:
                    # Keeping only the commits that has any of the word in fix_string_list in their commit.msg.
                    if any(fix_string in commit.msg for fix_string in fix_string_list):
                        
                        modifications=commit.modifications
                        
                        for modification in modifications:
                            
                            # Keeping only the code modifications of type MODIFY for java files.
                            if modification.change_type == ModificationType.MODIFY and modification.filename.endswith('.java'):
                                
                                # Keeping only the code modifications that are less than 100 lines of code.
                                code_diff = '\n'.join(modification.diff.splitlines())
                                if get_number_of_modified_lines(code_diff)<= 100:
                                    
                                    # Saving data only for commits with code modifications of type MODIFY and less than 100 lines of code.
                                    # data --> commit sha, commit message.
                                    sha=commit.hash
                                    msg=commit.msg
                                    # Deleting whitespace characters from commit.msg in order to save them in a more readable way to the csv.
                                    msglist = [item for item in msg.split()]
                                    msg = ' '.join(msglist)
                                    sha_outfile.write(sha + '\n')
                                    msg_outfile.write(msg + '\n')
                                    break
    except Exception as e:
        print('Error 404 !')
        print(e)                
                                
    print('Processing Done !!!')

if __name__ == "__main__":
          
    # Creating folder paths in which the data is saved.
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    if not os.path.exists(data_folder_path + "shas_files/"):
        os.makedirs(data_folder_path + "shas_files/")

    if not os.path.exists(data_folder_path + "msgs_files/"):
        os.makedirs(data_folder_path + "msgs_files/")

        
    # Reading the 3000 projects we want to analyse.
    with open('dataset_creation/repos3000Java.csv') as infile:
        lines= infile.readlines()
    repos = [line.split(';') for line in lines]
    repo_adresses = []
    for repo in repos:
        if int(repo[3])<2500:
            repo_adresses.append(repo[0])
    # Repos analysis.
    test = repo_adresses[2047:]
    parallel.pool_of_workers(4, process_repo, test)

