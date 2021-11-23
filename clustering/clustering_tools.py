import os
import numpy as np
import pandas as pd
import csv
import sys
import statistics
import xml.etree.ElementTree as ET
from tree_distance.tree_edit_distance.pq_grams.node_tree import ast_tree_to_node_tree
from tree_distance.tree_edit_distance.pq_grams.PyGram import Profile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset_creation.texthelpers import process_text


csv.field_size_limit(500000)


def cohesion(additions_deletions, centroid_additions_deletions, inserts_deletes_ast_trees, inserts_deletes_centroid_ast):
    """
    :param additions_deletions_ast_trees
    :param centroid_ast
    :return: the cohesion of a cluster
    """

    #profiles_additions_deletions = []
    profiles_inserts_deletes = []
        
    """ for ast_tree in additions_deletions_ast_trees:
        profiles_additions_deletions.append(Profile(ast_tree_to_node_tree(ET.fromstring(ast_tree))))

    centroid_additions_deletions_profile = Profile(ast_tree_to_node_tree(ET.fromstring(additions_deletions_centroid_ast))) """

    for ast_tree in inserts_deletes_ast_trees:
        profiles_inserts_deletes.append(Profile(ast_tree_to_node_tree(ET.fromstring(ast_tree))))

    centroid_inserts_deletes_profile = Profile(ast_tree_to_node_tree(ET.fromstring(inserts_deletes_centroid_ast)))
    
    cleaned_additions_deletions = list(map(process_text,additions_deletions))
           
    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(cleaned_additions_deletions)
    vectors = sparse_matrix.toarray()
    
    cleaned_centroid_additions_deletions = process_text(centroid_additions_deletions) 
    
    tgt_transform = vectorizer.transform([cleaned_centroid_additions_deletions]).toarray()
    tgt_cosine = cosine_similarity(vectors,tgt_transform)
    #print(tgt_cosine)
    
    
    #temp_additions_deletions = 0
    temp_inserts_deletes = 0
    temp_cos = 0

    for i in range(len(profiles_inserts_deletes)):

        #temp_additions_deletions = temp_additions_deletions + (1-centroid_additions_deletions_profile.edit_distance(profiles_additions_deletions[i]))
        #print(centroid_inserts_deletes_profile.edit_distance(profiles_inserts_deletes[i]))
        temp_inserts_deletes = temp_inserts_deletes + (1-centroid_inserts_deletes_profile.edit_distance(profiles_inserts_deletes[i]))
        temp_cos = temp_cos + float(tgt_cosine[i])
    temp_inserts_deletes -= 1
    temp_cos -= 1
    coh = ((temp_inserts_deletes + temp_cos) / 2)/(len(profiles_inserts_deletes) - 1 )

    return coh


def repos_per_cluster(data):
    """
    :param data
    :return: the number of repositories of a cluster
    """

    temp = []

    repo_names = data["repo_name"]

    for repo_name in repo_names:
        if repo_name not in temp:
            temp.append(repo_name)

    repositories = len(temp)

    return repositories


def clusters_selection(data, type_of_change, clusters, centroids, points_per_cluster, size_threshold, repos_threshold, cohesion_threshold):
    """
    After clustering process we choose only the clusters with size bigger than size_threshold

    :param centroids:
    :param clusters:
    :param data:
    :param points_per_cluster:
    :param size_threshold
    :param cohesion_threshold:
    :param repos_threshold:
    :return: the selected clusters
    """

    n_clusters = max(points_per_cluster, key=int)

    selected_clusters = []
    coh_selected_clusters = []

    for cluster in range(n_clusters):

        if points_per_cluster[cluster] >= size_threshold:

            points = np.where(clusters == cluster)

            cluster_data_frame = pd.DataFrame(columns=["sha", "filename","repo_name", "date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])

            for point in points[0]:
                cluster_data_frame = cluster_data_frame.append(data.loc[point])
            
            #cohesion(additions_deletions_ast_trees, additions_deletions_centroid_ast, additions_deletions, centroid_additions_deletions, inserts_deletes_ast_trees, inserts_deletes_centroid_ast):
            if type_of_change == "only_additions":
                coh = cohesion(cluster_data_frame["code_additions"],
                            data["code_additions"][int(centroids[cluster])],cluster_data_frame["code_inserts_ast"],
                            data["code_inserts_ast"][int(centroids[cluster])])
                if repos_per_cluster(cluster_data_frame) > repos_threshold and  coh >= cohesion_threshold:
                    selected_clusters.append(cluster)
                    coh_selected_clusters.append(coh)
            elif type_of_change == "only_deletions":
                coh = cohesion(cluster_data_frame["code_deletions"],
                            data["code_deletions"][int(centroids[cluster])],cluster_data_frame["code_deletes_ast"],
                            data["code_deletes_ast"][int(centroids[cluster])])
                if repos_per_cluster(cluster_data_frame) > repos_threshold and coh >= cohesion_threshold:
                    selected_clusters.append(cluster)
                    coh_selected_clusters.append(coh)
            else:
                coh = statistics.mean([cohesion(cluster_data_frame["code_additions"],
                            data["code_additions"][int(centroids[cluster])],cluster_data_frame["code_inserts_ast"],
                            data["code_inserts_ast"][int(centroids[cluster])]) , cohesion(cluster_data_frame["code_deletions"],
                            data["code_deletions"][int(centroids[cluster])],cluster_data_frame["code_deletes_ast"],
                            data["code_deletes_ast"][int(centroids[cluster])])])
                if repos_per_cluster(cluster_data_frame) > repos_threshold and coh >= cohesion_threshold:
                    selected_clusters.append(cluster)
                    coh_selected_clusters.append(coh)
            

    return selected_clusters,coh_selected_clusters


def compute_centroids(clusters, distance_matrix):
    """
    Compute the centroid for each cluster, choose as centroid the point with the smallest average distance
    from the other points of the cluster
    """

    n_clusters = max(clusters) + 1

    centroids = np.zeros(n_clusters)

    for cluster in range(n_clusters):

        # find the points and the size of each cluster
        cluster_points = np.where(clusters == cluster)[0]
        n_points = len(cluster_points)

        # initialization of the minimum sse (using a large value) and the centroid of te cluster (using the first point)
        min_sse = 1000000
        centroids[cluster] = cluster_points[0]

        for point in cluster_points:

            # average distance
            # print(distance_matrix[point, cluster_points])
            sse = sum(distance_matrix[point, cluster_points]) / n_points

            if sse < min_sse:

                min_sse = sse
                centroids[cluster] = point

    return centroids


def wss(clusters, distance_matrix):
    """
    Compute the within cluster sum of squares

    :param clusters:
    :param distance_matrix:
    :return:
    """

    n_clusters = max(clusters) + 1

    sse = 0

    centroids = compute_centroids(clusters, distance_matrix)

    for cluster in range(n_clusters):

        # find the points and the size of each cluster
        cluster_points = np.where(clusters == cluster)[0]
        n_points = len(cluster_points)

        temp = 0

        for point in cluster_points:
            # print(centroids[cluster])
            temp = temp + distance_matrix[point, int(centroids[cluster])]

        temp = temp/n_points

        sse = temp + sse

    sse = sse/n_clusters

    return sse
