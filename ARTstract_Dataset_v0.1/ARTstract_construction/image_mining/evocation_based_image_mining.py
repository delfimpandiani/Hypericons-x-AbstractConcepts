"""
Script Name: ARTstract Image Mining

Overview:
This script is designed to perform data mining operations for the ARTstract project. It retrieves image names from a CSV file,
filters images based on dataset names, checks their existence, renames and copies them to a destination folder, resizes the images,
and organizes them into cluster-specific datasets.

Functionality:
1. get_all_imgs_list(filepath): Reads a CSV file and extracts image names into a JSON file.
2. get_old_new_names_dict(filepath): Reads a CSV file and extracts old and new names into a JSON file.
3. manipulate_artemis_string(string): Manipulates the Artemis dataset image names to match the desired format.
4. to_lowercase(dataset_folder): Converts folder names within a dataset to lowercase.
5. get_dataset_ioi(all_imgs_list, dataset_name): Filters images belonging to a specific dataset and returns their names and paths.
6. get_all_dataset_files(directory_name): Returns a list of all file paths within a directory and its subdirectories.
7. check_existence(dataset_name, dataset_ioi_paths, dataset_all_files): Checks if the images of interest (IOI) from a dataset exist in the dataset.
8. mine_dataset_ioi(dataset_name, dataset_ioi, dataset_ioi_paths, old_new_dict): Copies and renames the images of interest to a destination folder.
9. resize_all_overall_images(): Resizes all images in the overall dataset.
10. get_cluster_lists_dict(filepath): Reads a CSV file and returns a dictionary of cluster names and corresponding image names.
11. create_cluster_dataset_folders(lists_dict): Creates folders for each cluster in the specified location.
12. mine_ioi_by_cluster(lists_dict): Moves images to corresponding cluster folders.

Usage:
1. Update the file paths and dataset names according to your specific setup.
2. Uncomment the desired operations in the execution section.

"""


import csv
import json
import os
import shutil
import cv2
import re
import urllib.request
import time

def get_all_imgs_list(filepath):
    """
    get_all_imgs_list() takes in a filepath of a CSV file as an input, reads the CSV file,
        extracts the image names from the first column of each row,
        stores all of the image names in a list and writes it to a json file.

        Parameters:
        filepath (str): The filepath of the CSV file that contains the image names

        Returns:
        all_imgs (list): List of all the image names

        Outputs:
        ../output/all_imgs.json: A json file containing all the image names
    """
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        all_imgs = []
        for row in reader:
            image_name = row[0]
            all_imgs.append(image_name)
    with open("helper_data_structures/all_imgs.json", 'w') as outfile:
        json.dump(all_imgs, outfile)
    return all_imgs

def get_old_new_names_dict(filepath):
    """
    get_old_new_names_dict() takes in a filepath of a CSV file as an input,
        reads the CSV file, extracts the old and new names from each row,
        stores them in a dictionary and writes it to a json file.

        Parameters:
        filepath (str): The filepath of the CSV file that contains the old and new names

        Returns:
        old_new_dict (Dict): A dictionary containing the old and new names

        Outputs:
        ../output/old_new_names_dict.json: A json file containing the dictioanry of old and new names
    """
    old_new_dict = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        all_imgs = []
        for row in reader:
            old_name = row[0]
            new_name = row[1]
            old_new_dict[old_name] = new_name
    with open("helper_data_structures/old_new_names_dict.json", 'w') as outfile:
        json.dump(old_new_dict, outfile)
    return old_new_dict

def manipulate_artemis_string(string):
    # images have the structure
    ## "artemis_Northern_Renaissance/hieronymus-bosch/tiptych-of-temptation-of-st-anthony-1506.jpg", 
    ## "artemis_Realism/vasily-surikov/head-of-boyarynya-morozova-study-1886.jpg",
    # and we need
    ## "../input/artemis_dataset/northern_renaissance/hieronymus-bosch_tiptych-of-temptation-of-st-anthony-1506.jpg", 
    ## "../input/artemis_dataset/realism/vasily-surikov_head-of-boyarynya-morozova-study-1886.jpg",
    new_string = '../input/' + re.sub(r"_", "_dataset/", string, count=1)
    head, tail = new_string.rsplit("/", 1) 
    new_string = "_".join([head, tail]) 
    new_string = new_string.lower()
    return new_string

def to_lowercase(dataset_folder): 
    for foldername in os.listdir(dataset_folder):
        current_folder = os.path.join(dataset_folder, foldername)
        if os.path.isdir(current_folder):
            new_folder = os.path.join(dataset_folder, foldername.lower())
            os.rename(current_folder, new_folder)
    return 

def get_dataset_ioi(all_imgs_list, dataset_name):
    """
    get_dataset_ioi() takes in a list of all image names and a dataset name as inputs,
        filters the images that belong to the specific dataset,
        and returns a list of the image names and their paths that belong to that dataset.

        Parameters:
        all_imgs_list (List[str]): A list of all image names
        dataset_name (str): The name of the dataset of interest

        Returns:
        Tuple[List[str], List[str]]: A tuple of two lists, first list contains the image names of the specific dataset and the second one contains their paths
    """
    dataset_ioi_paths = []
    dataset_ioi = []
    for img in all_imgs_list:
        if dataset_name in img:
            if dataset_name == "advise":
                img_path = "../input/"+dataset_name+"_dataset/"+img.split("_")[1].lower()
                dataset_ioi.append(img)
                dataset_ioi_paths.append(img_path)
            elif dataset_name == "artemis":
                img_path = manipulate_artemis_string(img)
                dataset_ioi.append(img)
                dataset_ioi_paths.append(img_path)
            elif dataset_name == "artpedia":
                f = open("Local_input_data/artpedia.json", encoding='utf-8')
                image_data = json.load(f)
                img_id = img.split('_')[-1]
                img_path = image_data[img_id]["img_url"]
                dataset_ioi.append(img)
                dataset_ioi_paths.append(img_path)
            elif dataset_name == "tate":
                img_path = "Local_input_data/"+dataset_name+"_dataset/"+img.split("_")[1].lower()
                img_path = img_path + ".jpg"
                #ioi: "tate_A00001"
                #ioi_path: "../input/tate_dataset/A00001.jpg"
                dataset_ioi.append(img)
                dataset_ioi_paths.append(img_path)
    return dataset_ioi, dataset_ioi_paths

def get_all_dataset_files(directory_name):
    """
    get_all_dataset_files() takes in a directory name as an input,
        and returns a list of all the file paths in that directory and its subdirectories.

        Parameters:
        directory_name (str): The name of the directory of interest

        Returns:
        List[str]: A list of all the file paths in that directory and its subdirectories
    """
    list_of_dir = os.listdir(directory_name)
    all_files = list()
    for entry in list_of_dir:
        full_path = os.path.join(directory_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_all_dataset_files(full_path)
        else:
            all_files.append(full_path)
    all_files = [s.lower() for s in all_files]
    return all_files

def check_existence(dataset_name, dataset_ioi_paths, dataset_all_files):
    """
    check_existence() takes in a list of image names and a list of all file
        paths as inputs, checks if the images of interest (ioi) from that dataset
        are actually present in the dataset, and returns a list of the present images.

        Parameters:
        dataset_ioi (List[str]): A list of the image names of interest
        dataset_all_files (List[str]): A list of all file paths in the dataset

        Returns:
        List[str]: A list of the NOT present image names in the dataset
    """
    present = []
    not_present = []
    for ioi_path in dataset_ioi_paths:
        # if dataset_name == "tate":
        #     ioi_path = ioi_path + ".jpg"
        if ioi_path in dataset_all_files:
            print(ioi_path, "is present in the dataset")
            present.append(ioi_path)
        else:
            print(ioi_path, "is NOT present in dataset")
            not_present.append(ioi_path)
    print("there are ", len(present), "ioi present")
    print("there are ", len(not_present), "ioi MISSING")
    return not_present

def mine_dataset_ioi(dataset_name, dataset_ioi, dataset_ioi_paths, old_new_dict):
    """
    mine_dataset_ioi() takes in a list of image names, their corresponding paths
        and a dictionary of old and new names as inputs,
        creates a folder to store the images if it doesn't exist,
        renames the images to their new names as specified in the dictionary,
        and copies the images to the created folder.

        Parameters:
        dataset_ioi (List[str]): A list of the image names of interest
        dataset_ioi_paths (List[str]): A list of the image paths of interest
        old_new_dict (Dict[str,str]): A dictionary containing the old and new names for the images

        Returns:
        None: The function does not return anything, it just copies the images to the destination folder
    """
    if not os.path.exists("../../Local_ARTstract_Dataset_v0.0/overall_dataset"):
        os.mkdir("../../Local_ARTstract_Dataset_v0.0/overall_dataset")
    destination_path = "../../Local_ARTstract_Dataset_v0.0/overall_dataset"
    failed = []
    succeed = []
    for x in range(len(dataset_ioi)):
        ioi_name = dataset_ioi[x]
        ioi_path = dataset_ioi_paths[x] 
        orig_name = ioi_name
        new_name = old_new_dict[orig_name]+".jpg"
        new_path = os.path.join(destination_path, new_name)
        if os.path.exists(new_path):
            succeed.append(ioi_path)
        else:
            if dataset_name == "artpedia":
                # for downloading images directly from url
                try:
                    urllib.request.urlretrieve(ioi_path, new_path)
                    succeed.append(ioi_path)
                except:
                    try:
                        print("sleeping for a sec to make another request for", ioi_path)
                        time.sleep(1)
                        urllib.request.urlretrieve(ioi_path, new_path)
                        succeed.append(ioi_path)
                    except:
                        print("could not find", ioi_path)
                        failed.append(ioi_path)
                        continue
            else:
                if os.path.exists(ioi_path):
                    shutil.copy2(ioi_path, new_path)
                    succeed.append(ioi_path)
                else:
                    failed.append(ioi_path)

    # print("For dataset", dataset_name, "SUCCEEDED", succeed)
    print("For dataset", dataset_name, "NUMBER SUCCEEDED", len(succeed))
    # print("For dataset", dataset_name, "FAILED", failed)
    print("For dataset", dataset_name, "NUMBER FAILED", len(failed))
    return 

def resize_all_overall_images():
    source_path = "../../Local_ARTstract_Dataset_v0.0/overall_dataset"
    succeed = []
    failed = []
    for img in os.scandir(source_path):
        if img.is_file():
            img_path = img.path
            img_name = img.name
            resized_img_path = "../../Local_ARTstract_Dataset_v0/resized_dataset/" + img_name
            if os.path.exists(resized_img_path):
                succeed.append(img_path)
            else:
                try:
                    img = cv2.imread(img.path)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width = img.shape[:2]
                    if width > height:
                        target_size = (512, int(512 * height / width))
                    else:
                        target_size = (int(512 * width / height), 512)
                    img = cv2.resize(img, target_size, interpolation = cv2.INTER_LINEAR)
                    cv2.imwrite(resized_img_path, img)
                    succeed.append(img_path)
                except:
                    print(img_path, " did not work")
                    failed.append(img_path)
    # print("For dataset", dataset_name, "SUCCEEDED", succeed)
    print("For resizing of overall images, NUMBER SUCCEEDED", len(succeed))
    # print("For dataset", dataset_name, "FAILED", failed)
    print("For resizing of overall images, NUMBER FAILED", len(failed))

##################### CLUSTER SPECIFIC MINING #####################
def get_cluster_lists_dict(filepath):
    """
    get_cluster_lists_dict() takes in a filepath and reads the csv file at that location and outputs a dictionary,
        where the keys are the cluster names and the values are a list of image names that evoke the cluster UNIQUELY.

        Parameters:
        filepath (str): The path of the csv file

        Returns:
        Dict: A dictionary where the keys are the cluster names and the values are a list of image names that evoke the cluster UNIQUELY.

    """
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        lists_dict = {}
        for row in reader:
            image_name = row[1]
            for i in range(2, 14):
                if row[i] == "True":
                    column_name = header[i]
                    if column_name not in lists_dict:
                        lists_dict[column_name] = []
                    lists_dict[column_name].append(image_name)
    with open("helper_data_structures/lists_dict.json", 'w') as outfile:
        json.dump(lists_dict, outfile)
    return lists_dict

def create_cluster_dataset_folders(lists_dict):
    """
    create_cluster_dataset_folders() takes in a dictionary where the keys are the cluster names and the values are lists of image names,
        and creates a folder for each cluster in the specified location.

        Parameters:
        lists_dict (Dict): A dictionary where the keys are the cluster names and the values are lists of image names

        Returns:
        None: The function doesn't return anything, it creates folders in the specified location
    """
    for cluster_name, img_list in lists_dict.items():
        if not os.path.exists(f"../../Local_ARTstract_Dataset_v0.0/cluster_specific_datasets/{cluster_name}_dataset"):
            os.makedirs(f"../../Local_ARTstract_Dataset_v0.0/cluster_specific_datasets/{cluster_name}_dataset")
    return

def mine_ioi_by_cluster(lists_dict):
    """
    mine_ioi_by_cluster() takes in a dictionary where the keys are the cluster names and the values are lists of image names,
        and moves the images to the corresponding cluster folders in the specified location.

        Parameters:
        lists_dict (Dict): A dictionary where the keys are the cluster names and the values are lists of image names

        Returns:
        None: The function doesn't return anything, it moves images to the corresponding cluster folders
    """
    for cluster_name, img_list in lists_dict.items():
        for img in img_list:
            destination_path = "../output/cluster_specific_datasets/"+ cluster_name+"_dataset/"+ img + ".jpg"
            source_path = "../output/resized_dataset/" + img + ".jpg"
            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
    return


# # GENERAL: EXECUTION
# filepath = "../../ARTstract_construction/image_selection/output_dataframes/UNIQUE_dataset_dataframe.csv"
# all_imgs_list = get_all_imgs_list(filepath)
# old_new_dict = get_old_new_names_dict(filepath)
# to_lowercase("Local_input_data/artemis_dataset")
# dataset_name_list = ["advise", "artemis", "artpedia", "tate"]
# for dataset_name in dataset_name_list:
#     dataset_ioi, dataset_ioi_paths = get_dataset_ioi(all_imgs_list, dataset_name)
#     # for datasets that are accessed from local copies
#     if dataset_name != "artpedia":
#         directory_name = "Local_input_data/"+dataset_name+"_dataset"
#         dataset_all_files = get_all_dataset_files(directory_name)
#         checks = check_existence(dataset_name, dataset_ioi_paths, dataset_all_files)
#     mine_dataset_ioi(dataset_name, dataset_ioi, dataset_ioi_paths, old_new_dict)
# resize_all_overall_images()
#
# # CLUSTER SPECIFIC MINING: EXECUTION
# lists_dict = get_cluster_lists_dict(filepath)
# create_cluster_dataset_folders(lists_dict)
# mine_ioi_by_cluster(lists_dict)



