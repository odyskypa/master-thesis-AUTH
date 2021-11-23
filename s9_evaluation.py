from evaluation.evaluation_tools import *

if __name__ == '__main__':
    
    #type_of_changes = ['both','only_additions','only_deletions']
    #type_of_changes = ['only_additions','only_deletions']
    type_of_changes = ['both']
    for toc in type_of_changes:
        finding_test_file_in_clusters(True,toc)
        final_results(True,toc)