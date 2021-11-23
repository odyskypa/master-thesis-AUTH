import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from properties import data_folder_path



def boxplots_and_histogramms(type_of_change,features_path):

    figure_path_boxplot_number_of_lines = data_folder_path + "figures/boxplots/" + type_of_change + '_number_of_lines_plot.png' 
    figure_path_boxplot_number_of_names = data_folder_path + "figures/boxplots/" + type_of_change + '_number_of_names_plot.png'
    figure_path_boxplot_number_of_AST_children= data_folder_path + "figures/boxplots/" + type_of_change + '_number_of_AST_children_plot.png'
   
    
    print("Creating boxplots and histograms for " + type_of_change + " changes.")
    
    features = pd.read_csv(features_path)
    
    # N_O_L_C
    fig = plt.figure(figsize=(20,12))
    plt.title('Box Plots of number of lines changed for ' + type_of_change + ' changes.')
    fig.suptitle(' Number of lines difference = # of lines added + # of lines deleted . ',fontsize=14, fontweight='bold')

    # Creating axes instance
    ax = fig.add_subplot(111)
    ax.boxplot(features['number_of_lines'], labels =[str(len(features)) + ' ' + type_of_change + ' changes. '])
    ax.set_ylabel('Number of lines changed.')
    fig.savefig(figure_path_boxplot_number_of_lines)
    plt.show()
    fig.clf()
    
    plt.close()
    
    # Number of AST Children
    fig = plt.figure(figsize=(20,12))
    plt.title('Box Plots of Number of AST Children for ' + type_of_change + ' changes.')
    #fig.suptitle(' Number of AST Children = # of lines added + # of lines deleted . ',fontsize=14, fontweight='bold')

    # Creating axes instance
    ax = fig.add_subplot(111)
    ax.boxplot(features['number_of_AST_children'], labels =[str(len(features)) + ' ' + type_of_change + ' changes. '])
    ax.set_ylabel('Number of AST_children.')
    fig.savefig(figure_path_boxplot_number_of_AST_children)
    plt.show()
    fig.clf()
    
    plt.close()
    
    
    # Number of AST Children
    fig = plt.figure(figsize=(20,12))
    plt.title('Box Plots of Number of names for ' + type_of_change + ' changes.')
    fig.suptitle(' Number of Names = # of <Simple_Name> </Simple_name>. in change AST . ',fontsize=14, fontweight='bold')

    # Creating axes instance
    ax = fig.add_subplot(111)
    ax.boxplot(features['number_of_names'], labels =[str(len(features)) + ' ' + type_of_change + ' changes. '])
    ax.set_ylabel('Number of Names.')
    fig.savefig(figure_path_boxplot_number_of_names)
    plt.show()
    fig.clf()
    
    plt.close()

if __name__ == "__main__":   
    
    type_of_changes = ['only_additions','only_deletions','both']
    
    for toc in type_of_changes:
        features_path = data_folder_path + 'features/' + toc + '_features.csv'
        boxplots_and_histogramms(toc,features_path)
    