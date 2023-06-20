"""
Script:evocation_based_image_selection.py
Author: Delfina Sol Martinez Pandiani
Date: June 2023

This script contains functions for mining and analyzing evocation data for a given dataset.
It takes in a list of dataset names and performs various operations such as
creating an evocation dataframe that maps images to abstract0-concepet-clusters,
 calculating co-occurrence data, identifying unique evoker images, and gathering basic and general statistics.

Functions:
- get_dataset_basic_stats(ARTstract_cluster_dict, dataset_name): Gathers basic statistics for cluster evocation in a specific dataset and outputs a CSV file containing the cluster names as rows with frequency and average strength information.

- get_evocation_df(basic_stats): Gathers co-occurrence statistics for cluster evocation from the basic statistics and returns a Pandas dataframe with cluster names as columns, images as rows, and True or False as value.

- get_cooccurrence_dict(evocation_df, dataset_name): Takes in an evocation dataframe and a dataset name, and returns a dictionary of co-occurring clusters with lists of image URIs that evoke both clusters.

- get_unique_evoker_images(evocation_df, dataset_name): Takes in an evocation dataframe and a dataset name, and returns a dictionary of images that uniquely evoke each cluster, where the keys are cluster names and the values are lists of image URIs.

- get_dataset_detailed_stats(basic_stats, unique_evoker_images, ARTstract_cluster_dict, dataset_name): Takes in basic statistics, unique evoker images, ARTstract cluster dictionary, and dataset name, and returns an updated version of the basic statistics dataframe with detailed information such as the number of unique evoker images and their average evocation strength.

- mine_cluster_evocation(dataset_name): Mines evocation data for a given dataset by calling several other functions and returns an evocation dataframe.

- get_stats_and_img_ids(dataset_name_list): Takes a list of dataset names and returns a Pandas dataframe containing detailed statistics and unique evoker image IDs for each cluster across all datasets. If the detailed statistics and unique evoker image IDs have not been calculated for a dataset, it calls the mine_cluster_evocation() function.

- get_stats(dataset_name_list): Takes in a list of dataset names and returns a dataframe containing the combined statistics of evocation strength and unique evoker images for each cluster across all datasets.

- get_complete_img_dict(dataset_name_list): Takes in a list of dataset names and returns a dictionary of all images in those datasets, where the keys are image URIs and the values are dictionaries of information about the images.

"""

import os
import json
import csv
import pandas as pd
import itertools
from helper_functions.input_data_loader import get_image_data, get_cluster_data, get_mismatches
from helper_functions.dataset_specific_dicts import get_img_dict, get_cluster_dict
from functools import reduce

def get_dataset_basic_stats(ARTstract_cluster_dict, dataset_name):
    """
    get_dataset_basic_stats() is a function that gathers basic statistics for cluster evocation in a specific dataset.
    It takes in the ARTstract dictionary that is cluster-focused (ARTstract_cluster_dict), and the name of the dataset (string),
    and outputs a CSV file containing the cluster names as rows, each of which has information regarding the frequency and
    average strength of evocation in the specific dataset. It also returns a list of rows in the output CSV.

        Parameters:
        ARTstract_cluster_dict: (dict) A dictionary where the keys are cluster names and the values are dictionaries
        containing information about the cluster's evoked images, such as the image's URI, evocation strength, and evocation context.
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'.

        Returns:
        rows: (list) list of rows in the output CSV.

        Outputs as CSV:
            - {dataset_name}/basic_stats, a table with cluster names as rows, each of which has information regarding the frequency,
            and average evocation strength of evoker images in the specific dataset.
    """
    rows = []
    dataset_evoker_images = dataset_name + "_evoker_images"
    for cluster_name in ARTstract_cluster_dict:
        row = []
        evoker_images = ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images]
        evoker_image_list = []
        evocation_strengths_list = []
        for evoker_image in evoker_images:
            evoker_image_list.append(evoker_image)
            evocation_strength = evoker_images[evoker_image]["evocation_strength"]
            evocation_strengths_list.append(evocation_strength)
        cluster_count = len(evoker_images)
        evocation_average = (sum(evocation_strengths_list))/(len(evocation_strengths_list))
        row.append(cluster_name)
        row.append(evocation_average)
        row.append(cluster_count)        
        row.append(evoker_image_list)
        rows.append(row)
        stat = (dataset_name + " has " + str(cluster_count), "images that evoke", cluster_name, "with an average evocation of", evocation_average)
        print(stat)
    col_names = ['cluster_name', (dataset_name+'_evocation_average'), (dataset_name+'_count_evoker_imgs'), (dataset_name+'_evoker_images')]
    with open('../output/'+dataset_name+'/basic_stats.csv', 'w') as f:
      writer = csv.writer(f)
      writer.writerow(col_names)
      writer.writerows(rows)
    return rows

def get_evocation_df(basic_stats):
    """
    get_evocation_df() is a function that gathers cooccurrence statistics for cluster evocation from the basic statistics.
    It takes in basic_stats (list of lists) and returns a Pandas dataframe with cluster names as columns, images as rows, and True or False as value.

        Parameters:
        basic_stats: (list of lists) list of basic statistics for cluster evocation in a specific dataset.

        Returns:
        evocation_df: (Pandas dataframe) A dataframe with cluster names as columns, images as rows, and True or False as value.
    """
    cluster_dict = {}
    for row in basic_stats:
        cluster_dict[row[0]] = row[3]
    evocation_df = pd.DataFrame(columns=cluster_dict.keys())
    for cluster, imgs in cluster_dict.items():
        for img in imgs:
            evocation_df.at[img, cluster] = True
    evocation_df.fillna(False, inplace=True)

    return evocation_df

def get_cooccurrence_dict(evocation_df, dataset_name):
    """
    get_cooccurrence_dict() is a function that takes in an evocation dataframe and a dataset name,
    and returns a dictionary of co-occurring clusters, where the keys are tuples of the form (cluster1, cluster2)
    and the values are lists of image URIs that evoke both clusters.

        Parameters:
        evocation_df: (Pandas dataframe) A dataframe with cluster names as columns, images as rows, and True or False as value.
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'.

        Returns:
        cooccurrence_dict: (dict) A dictionary where the keys are tuples of the form (cluster1, cluster2) and the values are lists of image URIs that evoke both clusters.
    """
    clusters = evocation_df.columns
    combinations = list(itertools.combinations(clusters, 2))
    cooccurrence_dict = {}
    for comb in combinations:
        cluster1, cluster2 = comb
        imgs = evocation_df.query(f"{cluster1} == True and {cluster2} == True").index.tolist()
        cooccurrence_dict[comb] = imgs
    return cooccurrence_dict

def get_unique_evoker_images(evocation_df, dataset_name):
    """
    get_unique_evoker_images() is a function that takes in an evocation dataframe and a dataset name,
    and returns a dictionary of images that uniquely evoke each cluster, where the keys are cluster names
    and the values are lists of image URIs that evoke only that cluster.

        Parameters:
        evocation_df: (Pandas dataframe) A dataframe with cluster names as columns, images as rows, and True or False as value.
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'.

        Returns:
        unique_evoker_images: (dict) A dictionary where the keys are cluster names and the values are lists of image URIs that evoke only that cluster.

        Outputs as JSON:
            - {dataset_name}/unique_evoker_images.json, containing the dictionary
    """
    unique_evoker_images = {}
    for cluster in evocation_df.columns:    
        query = ' and '.join([f'{col} == False' for col in evocation_df.columns if col != cluster])
        unique_evoker_images[cluster] = {}
        unique_evoker_images_list = evocation_df.query(query)[cluster].index.tolist()
        unique_evoker_images[cluster]["unique_evoker_images"] = unique_evoker_images_list
        unique_evoker_images[cluster]["len_unique_evoker_images"] = len(unique_evoker_images_list)
    with open("../output/"+dataset_name+"/unique_evoker_images.json", "w") as outfile:
        json.dump(unique_evoker_images, outfile)
    return unique_evoker_images

def get_dataset_detailed_stats(basic_stats, unique_evoker_images, ARTstract_cluster_dict, dataset_name):
    """
    get_dataset_detailed_stats() is a function that takes in a basic statistics dataframe, a dictionary of unique evoker images,
    an ARTstract cluster dictionary, and the name of the dataset, and returns an updated version of the basic statistics dataframe
    that includes detailed information such as the number of unique evoker images and their average evocation strength.

        Parameters:
        basic_stats: (list of lists) A list of lists containing basic statistics for each cluster, including the cluster name, average evocation strength, and number of evoker images for the specific dataset.
        unique_evoker_images: (dict) A dictionary where the keys are cluster names and the values are lists of image URIs that evoke only that cluster.
        ARTstract_cluster_dict: (dict) The ARTstract cluster dictionary containing information about each cluster, including evoker images and evocation strength.
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'.

        Returns:
        rows: (list of lists) An updated list of lists containing detailed statistics for each cluster, including the number of unique evoker images and their average evocation strength.

        Outputs as CSV:
            - {dataset_name}/detailed_stats, a table with cluster names as rows, each of which has information regarding the frequency, average strength, number of unique evoker images, and average evocation strength of unique evoker images in the specific dataset.
    """
    dataset_evoker_images = dataset_name + "_evoker_images"
    rows = basic_stats
    for cluster_name in unique_evoker_images:
        evoker_images = ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images]
        all_evoker_images = ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images]
        UNIQUE_evokers_list = unique_evoker_images[cluster_name]["unique_evoker_images"]
        count_UNIQUE_evoker_img = len(UNIQUE_evokers_list)
        UNIQUE_evocation_strengths_list = []
        for unique_img in UNIQUE_evokers_list:
            evocation_strength = evoker_images[unique_img]["evocation_strength"]
            UNIQUE_evocation_strengths_list.append(evocation_strength)
            if len(UNIQUE_evocation_strengths_list) != 0:
                UNIQUE_average_evocation_strength = (sum(UNIQUE_evocation_strengths_list))/(len(UNIQUE_evocation_strengths_list))
            else: 
                UNIQUE_average_evocation_strength = 0
        # # check that the evocation is being calculated correctly
        # unique_evocation_count = len(UNIQUE_evocation_strengths_list)
        # print('for', cluster_name, 'there should be', count_UNIQUE_evoker_img, "evocation counts, there is actually", unique_evocation_count)
        for row in rows:
            if row[0] == cluster_name:
                row.append(count_UNIQUE_evoker_img)
                row.append(UNIQUE_evokers_list)
                row.append(UNIQUE_average_evocation_strength)
    col_names = ['cluster_name', (dataset_name+'_evocation_average'), (dataset_name+'_count_evoker_imgs'), (dataset_name+'_evoker_images'), (dataset_name+'_count_UNIQUE_evoker_imgs'), (dataset_name+'_UNIQUE_evoker_images'), (dataset_name+'_UNIQUE_average_evocation')]
    with open("../output/"+dataset_name+"/detailed_stats.csv", 'w') as f:
      writer = csv.writer(f)
      writer.writerow(col_names)
      writer.writerows(rows)
    return rows

def mine_cluster_evocation(dataset_name):
    """
    mine_cluster_evocation() is a function that mines evocation data for a given dataset.

        Parameters:
        dataset_name (str): name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'.

        Returns:
        dataset_evocation_df: A dataframe with cluster names as columns, images as rows, and True or False as value.
        This function first calls several other functions to gather cluster data, image data, mismatches,
        and then creates dictionaries for image and cluster data. Then it calls several other functions to gather basic statistics,
        evocation dataframe, cooccurrence data, unique evoker images, and detailed statistics.
    """
    cluster_data = get_cluster_data('../Local_input_data/ACsof_interest/selected_clusters_v0.1.json')
    image_data = get_image_data('../Local_input_data/'+ dataset_name +'.json')
    mismatches = get_mismatches("../input/mistmatches/mismatches_v0.1.json")
    ARTstract_img_dict = get_img_dict(dataset_name, image_data, cluster_data, mismatches)
    ARTstract_cluster_dict = get_cluster_dict(dataset_name, image_data, cluster_data, mismatches)
    dataset_basic_stats = get_dataset_basic_stats(ARTstract_cluster_dict, dataset_name)
    dataset_evocation_df = get_evocation_df(dataset_basic_stats)
    dataset_cooccurrence_dict = get_cooccurrence_dict(dataset_evocation_df, dataset_name)
    dataset_unique_evoker_images = get_unique_evoker_images(dataset_evocation_df, dataset_name)
    dataset_detail_stats = get_dataset_detailed_stats(dataset_basic_stats, dataset_unique_evoker_images, ARTstract_cluster_dict, dataset_name)
    return dataset_evocation_df

def get_stats_and_img_ids(dataset_name_list):
    """
    get_stats_and_img_ids() is a function that takes a list of dataset names and returns a Pandas dataframe
    containing detailed statistics and unique evoker image IDs for each cluster, across all datasets in the list. I
    f the detailed statistics and unique evoker image IDs have not been calculated for a dataset,
    the function will call the mine_cluster_evocation() function to do so.

        Parameters:
        dataset_name_list: (list) A list of strings containing the names of the datasets for which statistics and unique evoker images are to be gathered.

        Returns:
        stats_and_img_ids_df: (Pandas dataframe) A dataframe with cluster names as columns, and statistics and unique evoker image IDs for each cluster, across all datasets in the list.


        Outputs as CSV:
            - output/combined_stats_and_img_ids.csv, a CSV file containing the concatenated statistics of each dataset.
    """
    concatenation_list = []
    for dataset_name in dataset_name_list:
        file_path = f"../output/{dataset_name}/detailed_stats.csv"
        if os.path.isfile(file_path):
            print(f"{file_path} exists.") 
        else:
            mine_cluster_evocation(dataset_name)
        df = pd.read_csv(file_path)
        new_df = df
        concatenation_list.append(df)
    stats_and_img_ids_df = reduce(lambda left,right: pd.merge(left,right,on='cluster_name', how='outer'), concatenation_list)
    stats_and_img_ids_df.to_csv('../output/combined_stats_and_img_ids.csv', index=False)
    return stats_and_img_ids_df

def get_stats(dataset_name_list):
    """
    get_stats() is a function that takes in a list of dataset names, and returns a dataframe containing the combined statistics of evocation strength and unique evoker images for each cluster across all datasets. It also saves the dataframe as a csv file.

        Parameters:
        dataset_name_list: (list) A list of strings containing the names of the datasets to be processed.

        Returns:
        stats_df: (Pandas dataframe) A dataframe containing the combined statistics of evocation strength and unique evoker images for each cluster across all datasets.

        Outputs:
         - output/combined_stats.csv, a csv file containing the dataframe.
    """
    file_path = f"../output/combined_stats_and_img_ids.csv"
    if os.path.isfile(file_path):
        print(f"{file_path} exists.")
    else:
        get_stats_and_img_ids(dataset_name_list)
    df = pd.read_csv(file_path)
    columns_to_drop = []
    for dataset_name in dataset_name_list:
        column_to_drop_1 = dataset_name + '_evoker_images' 
        column_to_drop_2 =  dataset_name + '_UNIQUE_evoker_images'
        columns_to_drop.append(column_to_drop_1)
        columns_to_drop.append(column_to_drop_2)
    stats_df = df.drop(columns_to_drop, axis=1)
    stats_df.to_csv('../output/combined_stats.csv', index=False)
    return stats_df
         
def get_complete_img_dict(dataset_name_list):
    """
    get_complete_img_dict() is a function that takes in a list of dataset names and returns a dictionary of all images in those datasets, where the keys are image URIs and the values are dictionaries of information about the images.

        Parameters:
        dataset_name_list: (list) A list of strings, where each string is the name of a dataset.

        Returns:
        complete_img_dict: (dict) A dictionary where the keys are image URIs and the values are dictionaries of information about the images.

        Outputs as JSON:
         - complete_ARTstract_img_dict.json, containing the complete ARTstract_img_dict.
    """
    complete_img_dict = {}
    for dataset_name in dataset_name_list:
        img_dict_path = f"../output/{dataset_name}/ARTstract_img_dict.json"
        img_dict = json.load(open(img_dict_path, encoding='utf-8'))
        complete_img_dict.update(img_dict)
    with open("../output/complete_ARTstract_img_dict.json", "w") as outfile:
        json.dump(complete_img_dict, outfile)

    return complete_img_dict

def get_complete_cluster_dict(dataset_name_list):
    """
    get_complete_cluster_dict() is a function that takes in a list of dataset names and returns a dictionary of all the clusters and their evoker images, where the keys are cluster names and the values are dictionaries containing cluster information and evoker images from each dataset.

        Parameters:
        dataset_name_list: (list) A list of strings containing the names of the datasets from which the image data is sourced.

        Returns:
        ARTstract_cluster_dict: (dict) A dictionary where the keys are cluster names and the values are dictionaries containing cluster information and evoker images from each dataset.

        Outputs as JSON:
         - complete_ARTstract_cluster_dict.json, containing the complete ARTstract_cluster_dict.

    """
    cluster_data = get_cluster_data('../input/selected_clusters.json')
    ARTstract_cluster_dict = {}
    for cluster in cluster_data:
        cluster_id = cluster["cluster_id"]
        cluster_name = cluster["cluster_name"]
        cluster_words = cluster["symbols"]
        cluster_pic_ids = []
        ARTstract_cluster_dict[cluster_name] = {}
        ARTstract_cluster_dict[cluster_name]["cluster_id"] = cluster_id
        ARTstract_cluster_dict[cluster_name]["cluster_name"] = cluster_name
        ARTstract_cluster_dict[cluster_name]["cluster_words"] = cluster_words
        ARTstract_cluster_dict[cluster_name]["evoker_images"] = {}
        for dataset_name in dataset_name_list:
            dataset_evoker_images = dataset_name + "_evoker_images"
            dict_path = f"../output/{dataset_name}/ARTstract_cluster_dict.json"
            cluster_dict = json.load(open(dict_path, encoding='utf-8'))
            dict_of_interest = cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images]
            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images] = dict_of_interest
    with open("../output/complete_ARTstract_cluster_dict.json", "w") as outfile:
        json.dump(ARTstract_cluster_dict, outfile)
    return ARTstract_cluster_dict

def mine_evocation_df(dataset_name):
    """
    mine_evocation_df() is a function that takes in a dataset name, reads in the cluster data, image data, and mismatches, creates a cluster dictionary, basic statistics, and an evocation dataframe for that dataset.

        Parameters:
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'.

        Returns:
        dataset_evocation_df: (Pandas dataframe) A dataframe with cluster names as columns, images as rows, and evocation strength as value.

    """
    cluster_data = get_cluster_data('../input/selected_clusters.json')
    image_data = get_image_data('../input/'+ dataset_name +'.json')
    mismatches = get_mismatches("../input/mismatches.json")
    ARTstract_cluster_dict = get_cluster_dict(dataset_name, image_data, cluster_data, mismatches)
    dataset_basic_stats = get_dataset_basic_stats(ARTstract_cluster_dict, dataset_name)
    dataset_evocation_df = get_evocation_df(dataset_basic_stats)
    return dataset_evocation_df

def get_dataset_dataframe(dataset_name_list):
    """
    get_dataset_dataframe() is a function that takes in a list of dataset names and
    returns a dataframe that contains all the evocation data for each dataset,
    with a new column added that gives the ARTstract_ID to each image.

        Parameters:
        dataset_name_list: (list) a list of dataset names from which the evocation data is sourced.

        Returns:
        merged_df: (Pandas Dataframe) A dataframe that contains all the evocation data for each dataset, with a new column added that gives the ARTstract_ID to each image.

        Output as CSV:
         - dataset_dataframe.csv, containing the dataframe.
    """
    evocation_dfs_list = []
    for dataset_name in dataset_name_list:
        evocation_df = mine_evocation_df(dataset_name)
        evocation_dfs_list.append(evocation_df)
    merged_df = pd.concat(evocation_dfs_list, axis=0, ignore_index=False)
    merged_df.insert(loc=0, column="ARTstract_ID", value=range(1, len(merged_df) +1))    
    merged_df.to_csv('../output/dataset_dataframe.csv', index=True)
    return merged_df

def get_UNIQUE_dataset_dataframe(dataset_name_list):
    """
    get_UNIQUE_dataset_dataframe() is a function that takes in a list of dataset names and returns a dataframe that contains all the evocation data for each dataset, with a new column added that gives the ARTstract_ID to each image.

        Parameters:
        dataset_name_list: (list) a list of dataset names from which the evocation data is sourced.

        Returns:
        merged_df: (Pandas Dataframe) A dataframe that contains all the evocation data for each dataset, but whose rows are images that only evoke ONE AC cluster.

        Output as CSV:
         - UNIQUE_dataset_dataframe.csv, containing the dataframe.
    """
    merged_df = get_dataset_dataframe(dataset_name_list)
    print(merged_df)
    print("i was able to get the complete dataset df")
    unique_cluster_df_list = []
    for cluster in merged_df.columns:    
        if cluster == "ARTstract_ID":
            print("I am skipping this column")
            continue
        else:
            query = ' and '.join([f'{col} == False' for col in merged_df.columns if (col != cluster and col != "ARTstract_ID")])
            print("about to create the merged df with the query", query)
            unique_cluster_df = merged_df.query(query)
            unique_cluster_df_list.append(unique_cluster_df)
            print(unique_cluster_df)
    unique_merged_df = pd.concat(unique_cluster_df_list, axis=0, ignore_index=False)
    unique_merged_df.to_csv('../output/UNIQUE_dataset_dataframe.csv', index=True)
    return unique_merged_df


# _____________________________________________________________
# Execution
# _____________________________________________________________
# dataset_name_list = ["artpedia", "advise", "artemis", "tate"]
# get_stats(dataset_name_list)
# get_complete_cluster_dict(dataset_name_list)
# get_complete_img_dict(dataset_name_list)
# get_dataset_dataframe(dataset_name_list)
# get_UNIQUE_dataset_dataframe(dataset_name_list)  
# _____________________________________________________________
# Finished in 346.4s in MacOs 2 GHz Dual-Core Intel Core i5
# _____________________________________________________________