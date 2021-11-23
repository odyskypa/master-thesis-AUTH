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

def process_repo(project_dir):
    
    data_folder_path = 'C:/Users/Odyskypa/Desktop/data1/'
    git_repo_path = os.path.join(data_folder_path, project_dir)
    sha_path = data_folder_path + "shas_files/" + str(project_dir) + "_shas.csv"
    msg_path = data_folder_path + "msgs_files/" + str(project_dir) + "_msgs.csv"
    
    try:
        total = GitRepository(git_repo_path).total_commits()
        print('Processing ' + str(total) + ' commits....')
        repomining = RepositoryMining(git_repo_path, only_modifications_with_file_types=['.java'])
        with open(sha_path,'w') as sha_outfile, open(msg_path,'w') as msg_outfile :
            for commit in repomining.traverse_commits():
                if commit.in_main_branch:
                    # Keeping only the commits that has any of the word in fix_string_list in their commit.msg.
                    
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
    data_folder_path = 'C:/Users/Odyskypa/Desktop/data1/'
    project_dirs = [project_dir for project_dir in os.listdir(data_folder_path)]
    parallel.pool_of_workers(4, process_repo, project_dirs)