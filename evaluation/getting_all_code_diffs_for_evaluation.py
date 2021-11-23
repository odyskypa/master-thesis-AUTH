import os
import sys
import pandas as pd
from pydriller import RepositoryMining
from pydriller.domain.commit import ModificationType
from dataset_creation.dataset_creation_tools import split_code_patch_single_change,get_number_of_modified_lines
from tree_distance import parallel

def extract_commits_for_evaluation(repo_sha):
    data_folder_path = 'C:/Users/Odyskypa/Desktop/data1/'
    
    repo_name = repo_sha[:-9]
    print("Extracting commits info for project: " + repo_name)
    git_repo_path = os.path.join(data_folder_path, repo_name)
    sha_path = os.path.join(data_folder_path + "shas_files/", repo_sha)
    commits_path = os.path.join(data_folder_path + "commits_files/", repo_name + "_commits.csv")

    with open(sha_path) as infile:
        shas = [line.strip() for line in infile if line]
    repomining = RepositoryMining(git_repo_path, only_commits=shas)

    print("Processing " + str(len(shas)) + " commits...")
    start = False
    commits = iter(enumerate(repomining.traverse_commits()))
    
    for c, commit in commits:
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
                    code_after = "\n".join(modification.source_code.splitlines()).encode('ascii', 'ignore').decode()
                    _, _, code_additions, code_deletions = split_code_patch_single_change(code_diff)
                    
                    res = ['\\x'+sha, filename, repo_name, date, message, code_diff, code_before, code_after, code_additions, code_deletions]
                    df = pd.DataFrame.from_records([res], columns=["sha", "filename", "repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after",\
                                "code_additions", "code_deletions"])
                    df = df.set_index("sha", drop=True)
                    if start == False:
                        df.to_csv(commits_path, sep='\t',encoding='utf-8')
                        start = True
                    else:
                        df.to_csv(commits_path, sep='\t', encoding='utf-8', header=None, mode="a")

if __name__ == "__main__":
       
    data_folder_path = 'C:/Users/Odyskypa/Desktop/data1/'
    if not os.path.exists(data_folder_path + "commits_files/"):
        os.makedirs(data_folder_path + "commits_files/")
    repo_shas = [repo_sha for repo_sha in os.listdir(data_folder_path + "shas_files/") if repo_sha.endswith("shas.csv")]   
    
    start0 = False
    for repo_sha in repo_shas:
        extract_commits_for_evaluation(repo_sha)
