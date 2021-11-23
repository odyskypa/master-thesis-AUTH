import xml.etree.ElementTree as ET
import numpy as np
import os
import tree_distance.parallel
import time
import matplotlib.pyplot as plt
from properties import data_folder_path
from sklearn import manifold
from scipy.spatial.distance import squareform
from functools import partial
from tree_distance.tree_edit_distance.pq_grams.PyGram import Profile
from tree_distance.tree_edit_distance.pq_grams.node_tree import ast_tree_to_node_tree


def pq_gram_distance(ast_profiles, tree_index):

    print(tree_index)

    vector = np.zeros(len(ast_profiles))

    for i in range(tree_index+1):
        vector[i] = ast_profiles[tree_index].edit_distance(ast_profiles[i])

    return vector

def pq_grams_profiles(ast_trees):
    

    """
    :param: flag: if flag is True names of methods are added to the calculation of distance.
    for the pq-gram algorithm, compute the Profile of each ast tree
    """

    # profiles = [Profile(ast_tree) for ast_tree in ast_trees]

    profiles = []

    for ast_tree in ast_trees:
        
        #print('1',ast_tree)
        #print('2',ET.fromstring(ast_tree))
        #print('3',ast_tree_to_node_tree(ET.fromstring(ast_tree)))
        #print('4',Profile(ast_tree_to_node_tree(ET.fromstring(ast_tree))))
        """ if flag:
            profiles.append(Profile(ast_tree_to_node_tree(ET.fromstring(ast_tree),True)))
        else: """
        profiles.append(Profile(ast_tree_to_node_tree(ET.fromstring(ast_tree))))

    return profiles

def plot_distance_matrix(distance_matrix):
    
    """
    Plot the distance matrix using multi-dimensional scaling technique
    """

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed")

    results = mds.fit(distance_matrix)

    coords = results.embedding_

    plt.scatter(
        coords[:, 0], coords[:, 1], marker='o'
    )

    plt.show()

def calculate_distance_matrix(ast_trees, sha = '', flag = False, k = 0):
    """
    Calculation of AST'S distance matrix, also saving it as .txt file.
    
    :param data_matrix_path --> file_path to save the distance matrix.
    :param ast_trees --> ASTs we want to calculate distance matrix for them.
    :param sha --> sha of commit we are processing. This is used when we compute distance matrix for a specific commit.
    :param flag --> when this flag is True we compute distance matrix for every change in a single commit.
    :param k --> is used when flag is True so we can know which exactly change has been calculated.
    :param text --> if text is True names of methods and variables are used in the calculation of the matrix.
    
    :return distance_matrix.
    """           
    # Setting the column of data that we want to compute distance matrix for.
    no_samples = len(ast_trees)
    ast_trees_range = range(no_samples)
    # Calculating AST profiles for the column above.
    profiles = pq_grams_profiles(ast_trees)
        
    func = partial(pq_gram_distance,profiles)
    
    t1 = time.time()
    outputs = tree_distance.parallel.pool_of_workers(4, func, ast_trees_range)    
    t2 = time.time()
      
    elapsed_time = t2 - t1
    
    print("distance matrix computation time: " + str(elapsed_time))

    # Creating square form of distance matrix.
    distance_matrix = np.array(outputs, dtype = 'float32')
    
    
    return distance_matrix

def find_zero_distance_indexes(distance_matrix):
    """
    
    Finding the indexes of distance matrix with asts that have zero distance (duplicates).
    
    :param data_matrix: numpy nd array with the values of distance matrix.
    
    :return final_zero_distance_indexes: List of tuples containing the codes with zero distance between them.
    
    """
    
    zero_distance_indexes = np.where(distance_matrix == 0)
    list_of_indexes= list(zip(zero_distance_indexes[0], zero_distance_indexes[1]))
    new_zero_distance_indexes = list()
        
    for indexes in list_of_indexes:
            
        # Discarding the diagonial zeros and the up-diagonial same values of zero.
        if ((indexes[1],indexes[0]) in new_zero_distance_indexes) or (indexes[0] == indexes[1]):
            continue
        else:
            new_zero_distance_indexes.append(indexes)
    
    iset = set([frozenset(s) for s in new_zero_distance_indexes])  # Convert to a set of sets
    final_zero_distance_indexes = []
    while(iset):                  # While there are sets left to process:
        nset = set(iset.pop())      # Pop a new set
        check = len(iset)           # Does iset contain more sets
        while check:                # Until no more sets to check:
            check = False
            for s in iset.copy():       # For each other set:
                if nset.intersection(s):  # if they intersect:
                    check = True            # Must recheck previous sets
                    iset.remove(s)          # Remove it from remaining sets
                    nset.update(s)          # Add it to the current set
        temp = sorted(nset)
        final_zero_distance_indexes.append(tuple(temp))  # Convert back to a list of tuples
    
    return(final_zero_distance_indexes)

def calculate_distance_matrix_same_commit(distance_matrix_path, ast_trees, sha = '', flag = False, k = 0):
    """
    Calculation of AST'S distance matrix, also saving it as .txt file.
    
    :param data_matrix_path --> file_path to save the distance matrix.
    :param ast_trees --> ASTs we want to calculate distance matrix for them.
    :param sha --> sha of commit we are processing. This is used when we compute distance matrix for a specific commit.
    :param flag --> when this flag is True we compute distance matrix for every change in a single commit.
    :param k --> is used when flag is True so we can know which exactly change has been calculated.
    :param text --> if text is True names of methods and variables are used in the calculation of the matrix.
    
    :return distance_matrix.
    """
       
    if flag :
        if not os.path.exists(distance_matrix_path):
            os.makedirs(distance_matrix_path)
        filename = distance_matrix_path +"/" + sha + "_" + str(k) + '.txt'
    elif sha == '':
        filename = distance_matrix_path + '.txt'
    else:
        if not os.path.exists(distance_matrix_path):
            os.makedirs(distance_matrix_path)    
        filename = distance_matrix_path +"/" + sha +'.txt'
    
    if not os.path.exists(filename):
        
           
        # Setting the column of data that we want to compute distance matrix for.
        no_samples = len(ast_trees)
        ast_trees_range = range(no_samples)
        # Calculating AST profiles for the column above.
        profiles = pq_grams_profiles(ast_trees)
        #func = partial(pq_gram_distance,profiles)
        distance_matrix = []
        t1 = time.time()
        for i in ast_trees_range:
            distance_matrix.append(pq_gram_distance(profiles,i).tolist())
             
        t2 = time.time()
        
        elapsed_time = t2 - t1
        
        print("distance matrix computation time: " + str(elapsed_time))

        # Creating square form of distance matrix.
        distance_matrix = np.array(distance_matrix, dtype = 'float32')
        distance_matrix = distance_matrix + distance_matrix.T
        
        with open(filename,"w") as f:
            f.write("\n".join(" ".join(map(str, x)) for x in distance_matrix))
        
        #plot_distance_matrix(distance_matrix)
    else:
        distance_matrix = np.loadtxt(filename, dtype=float)
    return distance_matrix