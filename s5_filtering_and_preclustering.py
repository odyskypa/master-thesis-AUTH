import os
import sys
import csv
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from properties import data_folder_path
from tree_distance import parallel
from preprocess.code_complexity import *
from numpy.lib.function_base import percentile

plt.style.use('seaborn-deep')
csv.field_size_limit(50000000)

def changes_filtering(type_of_change):
    if not os.path.exists(data_folder_path + "filtered_files/"):
        os.makedirs(data_folder_path + "filtered_files/")
    
    data_path = data_folder_path + "merged_changes/" + type_of_change + '/' + type_of_change + '_changes.csv'
    data = pd.read_csv(data_path, engine='python',index_col=0)
    
    features_path = data_folder_path + "features/"  + type_of_change + '_features.csv'
    features = pd.read_csv(features_path)
    
    # FILTERING FEATURES BASED ON IQR AND STATISTICS FROM DESCRIBED CSV ABOVE.
    if type_of_change == 'only_additions' :
        
        
        upper_nol = 8.5
        upper_AST_children = 6.5
        upper_number_of_names = 20.5
        
        features_to_be_deleted = features[(features['number_of_lines'] > upper_nol ) | (features['number_of_AST_children'] > upper_AST_children ) | (features['number_of_names'] > upper_number_of_names ) ].index
        print(len(features_to_be_deleted))
        #data7 = data.loc[features_to_be_deleted7,['code_additions','code_additions_ast']]
        
        outliers_indexes = features_to_be_deleted
        features.drop(outliers_indexes,inplace=True)
        features.reset_index(drop=True, inplace=True)
        features.to_csv(data_folder_path + "filtered_files/" + type_of_change + '_filtered_features.csv')
        filtered_described  = features.describe()
        filtered_described_df_path = data_folder_path + 'filtered_files/' + str(type_of_change) + '_filtered_features_described.csv'
        filtered_described.to_csv(filtered_described_df_path, header = True)
            
        
    elif  type_of_change == 'only_deletions' :
        
        upper_nol = 8.5
        upper_AST_children = 6.5
        upper_number_of_names = 20.5
        
        features_to_be_deleted = features[(features['number_of_lines'] > upper_nol ) | (features['number_of_AST_children'] > upper_AST_children ) | (features['number_of_names'] > upper_number_of_names ) ].index
        print(len(features_to_be_deleted))
        #data7 = data.loc[features_to_be_deleted7,['code_additions','code_additions_ast']]
        
        outliers_indexes = features_to_be_deleted
        features.drop(outliers_indexes,inplace=True)
        features.reset_index(drop=True, inplace=True)
        features.to_csv(data_folder_path + "filtered_files/" + type_of_change + '_filtered_features.csv')
        filtered_described  = features.describe()
        filtered_described_df_path = data_folder_path + 'filtered_files/' + str(type_of_change) + '_filtered_features_described.csv'
        filtered_described.to_csv(filtered_described_df_path, header = True)
        
    else:
        upper_nol = 16.5
        upper_AST_children = 13.5
        upper_number_of_names = 41.5
        
        
        features_to_be_deleted = features[(features['number_of_lines'] > upper_nol ) | (features['number_of_AST_children'] > upper_AST_children ) | (features['number_of_names'] > upper_number_of_names ) ].index
        print(len(features_to_be_deleted))
        
        outliers_indexes = features_to_be_deleted
        features.drop(outliers_indexes,inplace=True)
        features.reset_index(drop=True, inplace=True)
        features.to_csv(data_folder_path + "filtered_files/" + type_of_change + '_filtered_features.csv')
        filtered_described  = features.describe()
        filtered_described_df_path = data_folder_path + 'filtered_files/' + str(type_of_change) + '_filtered_features_described.csv'
        filtered_described.to_csv(filtered_described_df_path, header = True)
    
    print(len(outliers_indexes))
    if len(outliers_indexes) > 0 :
        data.drop(outliers_indexes, inplace=True)
        data.reset_index(drop=True, inplace=True)
    data.to_csv(data_folder_path + "filtered_files/" + type_of_change + '_changes.csv')
    
def features_description(type_of_change):
    """
    The function purpose is to create a feature description in order to choose what values will be filtered for each 
    type of change.

    :return: features described and IQR for every feature.
    """
    
    
    data_path = data_folder_path + "merged_changes/" + type_of_change + '/' + type_of_change + '_changes.csv'
    features_path = data_folder_path + "features/"  + type_of_change + '_features.csv'
    features = pd.read_csv(features_path)
    print(features.head(10))
    
    data = pd.read_csv(data_path, engine='python',index_col=0)
    data.shape
    
    # CREATING CORRELATION FIGURE FOR ALL FEATURES.
    
    figure_path_correlation = data_folder_path + "figures/correlation/" + type_of_change + '_correlation.png'
    fig = plt.figure(figsize=(20,12),facecolor='w')
    cor = features.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.title("Features Correlation for category: " + type_of_change + " Changes.")
    fig.savefig(figure_path_correlation,bbox_inches='tight',dpi=100)
    #plt.show()
    plt.close()
    
    """ figure_path_pairplot = data_folder_path + "figures/correlation/" + type_of_change + '_pairplot.png'
    fig = plt.figure(figsize=(20,12),facecolor='w')
    sns.pairplot(features)
    plt.title("Features Pairplot for category: " + type_of_change + " Changes.")
    fig.savefig(figure_path_pairplot,bbox_inches='tight',dpi=100)
    #plt.show()
    plt.close() """
    
    """ figure_path_boxplot = data_folder_path + "figures/correlation/" + type_of_change + '_boxplot.png'
    fig = plt.figure(figsize=(20,12),facecolor='w')
    sns.boxplot(data=features)
    plt.title("Features Boxplots for category: " + type_of_change + " Changes.")
    fig.savefig(figure_path_pairplot,bbox_inches='tight',dpi=100)
    #plt.show()
    plt.close() """
    
    
    # CHECKING FEATURES SHAPE.
    features.shape
    
    # CREATING CSV WITH DESCRIPTION OF FEATURES FOR EXAMINING.
    described  = features.describe()
    described_df_path = data_folder_path + 'features/' + str(type_of_change) + '_described.csv'
    described.to_csv(described_df_path, header = True)
    
    # PRINTING ALL UPPER AND LOWER THRESHOLDS FOR FEATURES BASED ON IQR.
    
    for column in features.columns:
        q25, q75 = percentile(features[column], 25), percentile(features[column], 75)
        iqr = q75 - q25
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr) + '\n')
        
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        
        # identify outliers
        outliers = [x for x in features[column] if x < lower or x > upper]
        print('Identified outliers for column ' + column + ': %d' % len(outliers) + '\n')
        print(lower,upper,'\n')
      
def preclustering(type_of_change,min_clusters,max_clusters,step):
       
    data_path = data_folder_path + '/filtered_files/' + str(type_of_change) + "_changes.csv"
    features_path = data_folder_path + '/filtered_files/' + str(type_of_change) + "_filtered_features.csv"
    preclustering_figures = data_folder_path + "figures/preclustering/"
    preclustering_path = data_folder_path + 'preclustering/clusters/'
    labels_df_path = data_folder_path + 'preclustering/' + str(type_of_change) + '_groupby_labels.csv'
    
    if not os.path.exists(preclustering_figures):
        os.makedirs(preclustering_figures)
    if not os.path.exists(preclustering_path):
        os.makedirs(preclustering_path)
        
    silh_figure_path_k_means = data_folder_path + "figures/preclustering/" + type_of_change + "_silh.png"
    sse_figure_path_k_means = data_folder_path + "figures/preclustering/" + type_of_change + "_sse.png"
    figure_path_k_means_data = data_folder_path + "figures/preclustering/" + type_of_change + "_data.png"    
    
    
    data = pd.read_csv(data_path, engine='python',index_col=0)
    features = pd.read_csv(features_path,index_col=0)
    
    """ if type_of_change == 'only_additions':
        n_components = 5
    elif type_of_change == 'only_deletions':
        n_components = 5
    else:
        n_components = 5 """
    
    pca = PCA(10)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    pca = pca.fit(scaled_features)
    features_pca = pca.transform(scaled_features)
    print(pca.explained_variance_ratio_)
    print (pca.explained_variance_ratio_.cumsum())
    df = pd.DataFrame(features_pca)
    print(df.shape)
    
    # CHECKING WHICH K IS SUITABLE IN THE THREE DIFFERENT GROUPS OF DATA: ONLY_ADDITIONS,ONLY_DELETIONS,BOTH.
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    
    """ x = range(min_clusters, max_clusters, step)
    sse = []
    silhouette_coefficients = []
    max_silhouette_coefficient = -1
    optimum_n_clusters = 0
    for k in x:
        print(k)
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        score = silhouette_score(df, kmeans.labels_)
        silhouette_coefficients.append(score)
        sse.append(kmeans.inertia_)
        if score > max_silhouette_coefficient:
            max_silhouette_coefficient = score
            optimum_n_clusters = k
    
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(15, 12))
    plt.suptitle('SSE for different values of k for: ' + type_of_change)
    plt.plot(x, sse,'bo-')
    plt.xticks(x)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    
    plt.savefig(sse_figure_path_k_means)
    #plt.show()
    plt.close()
    
    plt.figure(figsize=(15, 12))
    plt.suptitle('Silhouette Coefficient for different values of k for: ' + type_of_change)
    plt.plot(x, silhouette_coefficients,'bo-')
    plt.xticks(x)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")    
    
    plt.savefig(silh_figure_path_k_means)
    #plt.show()
    plt.close()
    
    kl = KneeLocator(x, sse, curve="convex", direction="decreasing")
    print("K-means elbow for : ",type_of_change," is: ", kl.elbow)
    print("Max Sihlouette coefficient for : ",type_of_change," has achieved for number of clusters =  ", optimum_n_clusters ) """
    # CREATING THE CLUSTERS AFTER CHECKING THE OPT K FOR K-MEANS.
    
    kmeans = KMeans( init="k-means++",n_clusters=9,n_init=10,max_iter=300,random_state=42)
    labels = kmeans.fit_predict(df)
    centroids = kmeans.cluster_centers_
    points_per_cluster = Counter(labels)
    print(points_per_cluster)
    
    # SAVING GROUPBY LABELS COUNTERS AND FEATURES WITH LABELS DF.
    features['labels'] = labels
    labels_df = features.groupby('labels').agg(['mean','min','max','count'])
    labels_df.to_csv(labels_df_path,header=True)
    
    features_with_labels_df_path = data_folder_path + 'preclustering/' + str(type_of_change) + '_features_with_labels.csv'
    features.to_csv(features_with_labels_df_path, header = True)
    
    
    fig,ax = plt.subplots(figsize=(15, 12))
    plt.suptitle('K-means Clusters of: ' + type_of_change)
    scatter = ax.scatter(features['cc'], features['number_of_lines'], marker = 'o', s = 70, c= labels.astype(float))
    legend1 = ax.legend(*scatter.legend_elements(),fancybox=True,loc='best',title = 'Labels')
    legend1.get_frame().set_alpha(0.3)
    ax.add_artist(legend1)
    ax.scatter(centroids[:,0],centroids[:,1],marker = '+',s =20,color = 'red')
    plt.xlabel("Cyclomatic Complexity Difference")
    plt.ylabel("Number of Lines Difference")
    plt.savefig(figure_path_k_means_data)
    #plt.show()
    plt.close()
    
    clusters_path = data_folder_path + 'preclustering/clusters/' + type_of_change + "/"

    if not os.path.exists(clusters_path):
        os.mkdir(clusters_path)
    elif len(os.listdir(clusters_path)) != 0:
        files = [file for file in os.listdir(clusters_path)]
        for f in files:
            os.remove(os.path.join(clusters_path, f))
    
    for i in set(labels):

        points = np.where(labels == i)

        file_path = data_folder_path + "/preclustering/clusters/" + type_of_change + "/" + type_of_change +'_'+ str(i) + ".csv"

        cluster_data_frame = pd.DataFrame(columns=["sha", "filename", "repo_name" ,"date", "message", "code_diff", "method_code_before", "method_code_after", "method_code_before_ast", "method_code_after_ast", \
                "code_additions", "code_deletions", "code_additions_ast", "code_deletions_ast","code_inserts_ast","code_deletes_ast"])

        for point in points[0]:
            cluster_data_frame = cluster_data_frame.append(data.loc[point])

        cluster_data_frame.reset_index(drop=True,inplace=True)
        cluster_data_frame.to_csv(file_path)
        print("Preclustering phase for cluster : " + str(i) + " has been completed .")
    
    print(" EVERYTHING DONE!!!")

def clustering_directories_creation(type_of_changes):
    """
    files_organizer method creates the appropriate directories for the clustering
    and idioms formatting process
    """

    folders_names = ["clusters", "centroids", "distance_matrices_additions_deletions","distance_matrices_inserts_deletes","cosine_distance_matrices", "my_idioms", "figures"]
    if not os.path.exists(data_folder_path + 'clustering/'):
        os.makedirs(data_folder_path + 'clustering/')

    for folder in folders_names:
        if not os.path.exists(data_folder_path + 'clustering/' + folder + '/'):
            os.makedirs(data_folder_path + 'clustering/' + folder + '/')

        for toc in type_of_changes:
            if not os.path.exists(data_folder_path + 'clustering/' + folder + '/' + toc + '/'):
                os.makedirs(data_folder_path + 'clustering/' + folder + '/' + toc + '/')

if __name__ == "__main__":   
    
    #type_of_changes = ['only_additions','only_deletions','both']
    type_of_changes = ['both']
    clustering_directories_creation(type_of_changes)
    for toc in type_of_changes:
        features_description(toc)
        #changes_filtering(toc)
        #preclustering(toc,1,15,1)
        