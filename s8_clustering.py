from dataset_creation.distance_matrix import calculate_distance_matrix,plot_distance_matrix
import os
from collections import Counter
from statistics import mean
from joblib import load
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from properties import data_folder_path
from clustering.clustering_tools import clusters_selection, compute_centroids
from clustering.hierarchical_clustering import ag_hierarchical,plot_dendrogram
plt.style.use('seaborn-deep')


def clustering_analysis(data_file_name, type_of_change, min_clusters , max_clusters, step ,size_threshold,repos_threshold, cohesion_threshold):
    """
    Complete clustering analysis using silhouette coefficient to determine the optimum number of clusters,
    compute the centroids and make a selection of the best clusters based on a number of parameters

    :param data_file_name:
    :param type_of_blocks:
    :param min_clusters:
    :param max_clusters:
    :param step:
    :param repos_threshold:
    :param cohesion_threshold:
    :param size_threshold:
    :return:
    """

    # remove suffix
    data_file_name_without_suffix = data_file_name.split('.')
    data_file_name_without_suffix = data_file_name_without_suffix[0]

    # load the cleaned corpus
    #data_path = data_folder_path + "preclustering/" + type_of_change + "/" + data_file_name_without_suffix + "_Cleaned.csv"
    data_path = data_folder_path + "preclustering/clusters/" + type_of_change + "/" + data_file_name_without_suffix + ".csv"
    data = pd.read_csv(data_path,index_col= [0] )

    print(data)

    
    """ distance_matrix_additions_deletions_path = data_folder_path + "clustering/distance_matrices_additions_deletions/" + type_of_change + "/" + \
                        data_file_name_without_suffix + ".joblib" """
    
    distance_matrix_inserts_deletes_path = data_folder_path + "clustering/distance_matrices_inserts_deletes/" + type_of_change + "/" + \
                        data_file_name_without_suffix + ".joblib"
    
    cosine_distance_matrix_path = data_folder_path + "clustering/cosine_distance_matrices/" + type_of_change + "/" \
                        + data_file_name_without_suffix +'.joblib'

    """ distance_matrix_additions_deletions = load(distance_matrix_additions_deletions_path)

    distance_matrix_additions_deletions = squareform(distance_matrix_additions_deletions) """
    
    distance_matrix_inserts_deletes = load(distance_matrix_inserts_deletes_path)

    distance_matrix_inserts_deletes = squareform(distance_matrix_inserts_deletes)
    
    cosine_distance_matrix = load(cosine_distance_matrix_path)

    cosine_distance_matrix = squareform(cosine_distance_matrix)
    
    # Adding the two matrices and normalizing them.
    distance_matrix_final = np.add(distance_matrix_inserts_deletes,cosine_distance_matrix)
    distance_matrix_final = MinMaxScaler().fit_transform(distance_matrix_final)
    #plot_distance_matrix(distance_matrix_final)
    

    print(distance_matrix_final.dtype)
    print(distance_matrix_final.nbytes)

    print(distance_matrix_final.shape)
    
    x = range(min_clusters, max_clusters, step)

    max_silhouette_coefficient = -1

    silhouette_average_scores = []

    optimum_n_clusters = 0

    for n_clusters in x:

        print(n_clusters)

        clustering = ag_hierarchical('precomputed', 'average', n_clusters, distance_matrix_final)

        clusters = clustering.labels_

        sil_sampl = silhouette_samples(distance_matrix_final, clusters, metric='precomputed')

        average_silh = mean(sil_sampl)

        silhouette_average_scores.append(average_silh)

        if average_silh > max_silhouette_coefficient:
            max_silhouette_coefficient = average_silh
            optimum_n_clusters = n_clusters
    
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(15, 12))
    ax1 = plt.axes()

    ax1.axhline(y=max_silhouette_coefficient, color="red", linestyle="--")
    plt.plot(x, silhouette_average_scores, 'bo-')
    plt.title("Average Silhouette Score for file : " + data_file_name_without_suffix)
    plt.xlabel("Number Of Clusters", fontsize=12)
    plt.ylabel("Average Silhouette Score", fontsize=12)

    if not os.path.exists(data_folder_path + "clustering/figures/" + type_of_change + "/"):
        os.mkdir(data_folder_path + "clustering/figures/" + type_of_change + "/")

    figure_path = data_folder_path + "clustering/figures/" + type_of_change + "/" + data_file_name_without_suffix + ".jpg"
    plt.savefig(figure_path,bbox_inches='tight',dpi=100)

    #plt.show()

    print("optimum number of clusters:" + str(optimum_n_clusters))

    print("max silhouette coefficient:" + str(max_silhouette_coefficient))

    clustering = ag_hierarchical('precomputed', 'average', optimum_n_clusters, distance_matrix_final)
    #plot_dendrogram(distance_matrix, 'average')
    clusters = clustering.labels_
    print(optimum_n_clusters)

    points_per_cluster = Counter(clusters)

    print(points_per_cluster)

    # This function returns the centroid of each cluster (index), based on the sse of all points in a cluster
    # with the distance of them.
    centroids = compute_centroids(clusters, distance_matrix_final)

    # selection of the clusters using a size threshold
    selected_clusters,coh_selected_clusters = clusters_selection(data, type_of_change, clusters, centroids, points_per_cluster, size_threshold,
                                           repos_threshold, cohesion_threshold)
    

    print(selected_clusters)
    print(coh_selected_clusters)

    clusters_path = data_folder_path + 'clustering/clusters/' + type_of_change + '/' + data_file_name_without_suffix + '/'

    if not os.path.exists(clusters_path):
        os.mkdir(clusters_path)
    elif len(os.listdir(clusters_path)) != 0:
        files = [file for file in os.listdir(clusters_path)]
        for f in files:
            os.remove(os.path.join(clusters_path, f))

    centroids_path = data_folder_path + 'clustering/centroids/' + type_of_change + '/' + data_file_name_without_suffix + '/'

    if not os.path.exists(centroids_path):
        os.mkdir(centroids_path)
    elif len(os.listdir(centroids_path)) != 0:
        files = [file for file in os.listdir(centroids_path)]
        for f in files:
            os.remove(os.path.join(centroids_path, f))

    for c,i in enumerate(selected_clusters):

        centroids_data_frame = pd.DataFrame(columns=["sha", "filename","repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])

        centroids_data_frame = centroids_data_frame.append(data.loc[int(centroids[i])])

        points = np.where(clusters == i)

        file_path = data_folder_path + "/clustering/clusters/" + type_of_change + "/" + data_file_name_without_suffix \
                    + "/" + str(i) + "_coh = " + str(coh_selected_clusters[c]) + ".csv"

        cluster_data_frame = pd.DataFrame(columns=["sha", "filename", "repo_name" ,"date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])

        for point in points[0]:
            cluster_data_frame = cluster_data_frame.append(data.loc[point])

        cluster_data_frame.to_csv(file_path)

        centroids_path = data_folder_path + "clustering/centroids/" + type_of_change + "/" + data_file_name_without_suffix \
                         + "/" + str(i) + ".csv"

        centroids_data_frame.to_csv(centroids_path)

if __name__ == "__main__":
          
    print('Final Clustering : ')
    
    # ONLY ADDITIONS
    
    #clustering_analysis("only_additions_0.csv", "only_additions", 200, 6496, 25, 5, 3, 0.5)
    #clustering_analysis("only_additions_1.csv", "only_additions", 1000, 13032, 25, 5, 3, 0.5)
    #clustering_analysis("only_additions_2.csv", "only_additions", 200, 1614, 25, 5, 3, 0.5)
    #clustering_analysis("only_additions_3.csv", "only_additions", 200, 3286, 25, 5, 3, 0.5)
    #clustering_analysis("only_additions_4.csv", "only_additions", 5, 858, 25, 5, 3, 0.5)
    
    # ONLY DELETIONS
    
    #clustering_analysis("only_deletions_0.csv", "only_deletions", 20, 325, 5, 5, 3, 0.5)
    #clustering_analysis("only_deletions_1.csv", "only_deletions", 200, 1466, 25, 5, 3, 0.5)
    #clustering_analysis("only_deletions_2.csv", "only_deletions", 20, 532, 5, 5, 3, 0.5)
    #clustering_analysis("only_deletions_3.csv", "only_deletions", 200, 4672, 25, 5, 3, 0.5)
    #clustering_analysis("only_deletions_4.csv", "only_deletions", 200, 1126, 25, 5, 3, 0.5)
    
    # BOTH
    
    #clustering_analysis("both_0.csv", "both", 500, 2276, 50, 5, 5, 0.5)
    #clustering_analysis("both_1.csv", "both", 1000, 35000, 500, 5, 5, 0.5)
    #clustering_analysis("both_2.csv", "both", 500, 4026, 50, 5, 5, 0.5)
    #clustering_analysis("both_3.csv", "both", 500, 7525, 100, 5, 5, 0.5)
    #clustering_analysis("both_4.csv", "both", 500, 9272, 200, 5, 5, 0.5)
    clustering_analysis("both_5.csv", "both", 500, 12496, 400, 5, 5, 0.5)
    #clustering_analysis("both_6.csv", "both", 500, 6732, 100, 5, 5, 0.5)
    #clustering_analysis("both_7.csv", "both", 500, 4364, 50, 5, 5, 0.5)
    #clustering_analysis("both_8.csv", "both", 500, 5174, 75, 5, 5, 0.5)
    
    
    