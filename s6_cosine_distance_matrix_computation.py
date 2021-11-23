from properties import data_folder_path
import os
import pandas as pd
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from joblib import load,dump
from scipy.spatial.distance import squareform
import nltk
from dataset_creation.texthelpers import process_text
nltk.download('stopwords')
stopwords = stopwords.words('english')

def cosine_distance_computation(data_file_name, type_of_change):
    
    data_path = data_folder_path + "/preclustering/clusters/" + type_of_change + '/' + data_file_name
    data = pd.read_csv(data_path,index_col=0)
    print(data.shape)

    # remove suffix
    data_file_name_without_suffix = data_file_name.split('.')[0]

    if type_of_change == "only_additions":
               
        
        cosine_distance_matrix_path = data_folder_path + "clustering/cosine_distance_matrices/" + type_of_change + "/" \
                           + data_file_name_without_suffix +'.joblib'
        
        if not os.path.exists(cosine_distance_matrix_path):
            
            changes = data["code_additions"]
            
            cleaned_changes = list(map(process_text,changes))
            
            vectorizer = TfidfVectorizer().fit_transform(cleaned_changes)
            vectors = vectorizer.toarray()
            
            csim = cosine_similarity(vectors)
            cdis = 1 - csim

            np.fill_diagonal(cdis, 0)
            cdis = np.float32(cdis)
            cosine_distance_matrix = squareform(cdis)

            dump(cosine_distance_matrix, cosine_distance_matrix_path)
            
        else:
            cosine_distance_matrix = load(cosine_distance_matrix_path)

        cosine_distance_matrix = squareform(cosine_distance_matrix)
        
    elif type_of_change == "only_deletions":
        
        cosine_distance_matrix_path = data_folder_path + "clustering/cosine_distance_matrices/" + type_of_change + "/" \
                           + data_file_name_without_suffix +'.joblib'
        
        if not os.path.exists(cosine_distance_matrix_path):
            
            changes = data["code_deletions"]
            
            cleaned_changes = list(map(process_text,changes))
            
            vectorizer = TfidfVectorizer().fit_transform(cleaned_changes)
            vectors = vectorizer.toarray()
            
            csim = cosine_similarity(vectors)
            cdis = 1 - csim
            
            np.fill_diagonal(cdis, 0)
            cdis = np.float32(cdis)
            cosine_distance_matrix = squareform(cdis)

            dump(cosine_distance_matrix, cosine_distance_matrix_path)
        else:
            cosine_distance_matrix = load(cosine_distance_matrix_path)

        cosine_distance_matrix = squareform(cosine_distance_matrix)
        
    else:
        
        add_cosine_distance_matrix_path = data_folder_path + "clustering/cosine_distance_matrices/" + type_of_change + "/" \
                           + data_file_name_without_suffix +'_additions.joblib'
        
        del_cosine_distance_matrix_path = data_folder_path + "clustering/cosine_distance_matrices/" + type_of_change + "/" \
                           + data_file_name_without_suffix +'_deletions.joblib'
        
        cosine_distance_matrix_path = data_folder_path + "clustering/cosine_distance_matrices/" + type_of_change + "/" \
                           + data_file_name_without_suffix +'.joblib'
        
        if not os.path.exists(add_cosine_distance_matrix_path):
                           
            add_changes = data["code_additions"]
            
            cleaned_add_changes = list(map(process_text,add_changes))

            add_vectorizer = TfidfVectorizer().fit_transform(cleaned_add_changes)
            add_vectors = add_vectorizer.toarray()
            add_csim = cosine_similarity(add_vectors)
            
            add_cdis = 1 - add_csim
            
            np.fill_diagonal(add_cdis, 0)
            add_cdis = np.float32(add_cdis)
            add_cosine_distance_matrix = squareform(add_cdis)

            dump(add_cosine_distance_matrix, add_cosine_distance_matrix_path)
        else:
            
            add_cosine_distance_matrix = load(add_cosine_distance_matrix_path)

        add_cosine_distance_matrix = squareform(add_cosine_distance_matrix)
        
        if not os.path.exists(del_cosine_distance_matrix_path):
                           
            del_changes = data["code_deletions"]
            
            cleaned_del_changes = list(map(process_text,del_changes))
            
            del_vectorizer = TfidfVectorizer().fit_transform(cleaned_del_changes)
            del_vectors = del_vectorizer.toarray()
            del_csim = cosine_similarity(del_vectors)
            
            del_cdis = 1 - del_csim
            
            np.fill_diagonal(del_cdis, 0)
            del_cdis = np.float32(del_cdis)
            del_cosine_distance_matrix = squareform(del_cdis)
            
            dump(del_cosine_distance_matrix, del_cosine_distance_matrix_path)
        else:
            
            del_cosine_distance_matrix = load(del_cosine_distance_matrix_path)

        del_cosine_distance_matrix = squareform(del_cosine_distance_matrix)
        
        if not os.path.exists(cosine_distance_matrix_path):
            
            cosine_distance_matrix = np.add(add_cosine_distance_matrix, del_cosine_distance_matrix)
            cosine_distance_matrix = squareform(cosine_distance_matrix)

            dump(cosine_distance_matrix, cosine_distance_matrix_path)
        else:
            cosine_distance_matrix = load(cosine_distance_matrix_path)
        
        cosine_distance_matrix = squareform(cosine_distance_matrix)      

if __name__ == "__main__":
    
    type_of_changes = ['only_additions','only_deletions','both']
    for toc in type_of_changes:
        csv_names = []
        for csv_name in os.listdir(data_folder_path + 'preclustering/clusters/' + toc):
            csv_names.append(csv_name)
        for csv_name in csv_names:
            cosine_distance_computation(csv_name, toc)
    print(' EVERYTHING DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')