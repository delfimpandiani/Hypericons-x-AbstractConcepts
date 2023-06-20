"""
This script creates dictionaries for the identification of images in art datasets that may evoke certain abstract concepts.

The main purpose of this script is to generate dictionaries that contain knowledge about the evocation of abstract concepts from images.
 It contains two functions, 'get_img_dict()' and 'get_cluster_dict()', which can be used to create dictionaries with different focuses.

- 'get_img_dict()' creates a dictionary where the keys represent image IDs, and the values contain information specific
to each image, such as feature vectors, metadata, etc.

- 'get_cluster_dict()' creates a dictionary where the keys represent abstract-concept-cluster IDs,
 and the values contain information specific to each cluster, such as representative images, statistics, etc.

Usage:
1. Import the script into your project.
2. Call the desired function(s) to generate the dictionaries based on your requirements.

Note:
- The dictionaries generated by this script can be further utilized for various tasks, such as visualization, analysis, etc.
"""

import json

def get_img_dict(dataset_name, image_data, cluster_data, mismatches):
    """
    get_img_dict(dataset_name:str, image_data:dict cluster_data:list, mismatches:dict) -> Dict:
    get_img_dict() is a function that creates a dictionary of image data, where the keys are image URIs and the values are dictionaries containing information about the image's evoked clusters, such as the cluster's name, words, evocation strength, and evocation context.

        Parameters:
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'
        image_data: (dict) a dictionary of image data, where the keys are image IDs and the values are dictionaries of image information
        cluster_data: (list) a list of dictionaries, each containing information about a single cluster, such as its ID and name
        mismatches: (dict) a dictionary where keys are cluster names and the values are lists of words that should not be matched as evoking the cluster

        Returns:
        ARTstract_img_dict: (dict) a dictionary where the keys are image URIs and the values are dictionaries containing information about the image's evoked clusters, such as the cluster's name, words, evocation strength, and evocation context.

        Outputs as CSV:
            - {dataset_name}/ARTstract_img_dict, containing the dictionary ARTstract_img_dict
    """
    ARTstract_img_dict = {}
    for cluster in cluster_data:
        cluster_id = cluster["cluster_id"]
        cluster_name = cluster["cluster_name"]
        cluster_words = cluster["symbols"]
        cluster_pic_ids = []
        for key, value in image_data.items():
            evocation_strength = 0
            key_URI = dataset_name + '_' + key 
            evocation_context = []
            #ARTPEDIA
            if dataset_name == 'artpedia':
                for item, item_value in value.items():
                    if item == 'visual_sentences':
                        for cluster_word in cluster_words:
                            for visual_sentence in item_value:
                                for visual_word in visual_sentence.split():
                                    if visual_word not in mismatches[cluster_name]:
                                        if cluster_word in visual_word:
                                            if key_URI not in ARTstract_img_dict.keys():
                                                ARTstract_img_dict[key_URI] = {}
                                            ARTstract_img_dict[key_URI]["source_dataset"] = dataset_name
                                            ARTstract_img_dict[key_URI]["source_id"] = key
                                            if "evoked_clusters" not in ARTstract_img_dict[key_URI].keys():
                                                ARTstract_img_dict[key_URI]["evoked_clusters"] = {}
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id] = {}
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_name"] = cluster_name
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_words"] = cluster_words
                                            if key not in cluster_pic_ids:
                                                evocation_strength = 1
                                                cluster_pic_ids.append(key)
                                                evocation_context.append(visual_sentence)
                                            else:
                                                if visual_sentence not in evocation_context:
                                                    evocation_context.append(visual_sentence)
                                                    evocation_strength += 1
                                                    print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_strength"] = evocation_strength
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_context"] = evocation_context
            #ADVISE
            elif dataset_name == 'advise':
                for annotation in value:
                    for item in annotation:
                        if isinstance(item, str): #if the item is a string
                            symbol_annotation = item.lower().replace('/',' ').split()
                            for cluster_word in cluster_words:
                                for symbol_word in symbol_annotation:
                                    if symbol_word not in mismatches[cluster_name]:
                                        if cluster_word in symbol_word:
                                            if key_URI not in ARTstract_img_dict.keys():
                                                ARTstract_img_dict[key_URI] = {}
                                                ARTstract_img_dict[key_URI]["source_dataset"] = dataset_name
                                                ARTstract_img_dict[key_URI]["source_id"] = key
                                            if "evoked_clusters" not in ARTstract_img_dict[key_URI].keys():
                                                ARTstract_img_dict[key_URI]["evoked_clusters"] = {}
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id] = {}
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_name"] = cluster_name
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_words"] = cluster_words
                                            if key not in cluster_pic_ids:
                                                evocation_strength = 1
                                                cluster_pic_ids.append(key)
                                                evocation_context.append(item)
                                            else:
                                                if item not in evocation_context:
                                                    evocation_context.append(item)
                                                    evocation_strength += 1
                                                    print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_strength"] = evocation_strength
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_context"] = evocation_context                         
            #ARTEMIS
            elif dataset_name == 'artemis':
                for annotation_dict in value:
                    for item, item_value in annotation_dict.items():
                        if item == 'utterance':
                            visual_sentence = item_value
                            for cluster_word in cluster_words:
                                for visual_word in visual_sentence.split():
                                    if visual_word not in mismatches[cluster_name]:
                                        if cluster_word in visual_word:
                                            if key_URI not in ARTstract_img_dict.keys():
                                                ARTstract_img_dict[key_URI] = {}
                                            ARTstract_img_dict[key_URI]["source_dataset"] = dataset_name
                                            ARTstract_img_dict[key_URI]["source_id"] = key
                                            if "evoked_clusters" not in ARTstract_img_dict[key_URI].keys():
                                                ARTstract_img_dict[key_URI]["evoked_clusters"] = {}
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id] = {}
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_name"] = cluster_name
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_words"] = cluster_words
                                            if key not in cluster_pic_ids:
                                                evocation_strength = 1
                                                cluster_pic_ids.append(key)
                                                evocation_context.append(visual_sentence)
                                            else:
                                                if visual_sentence not in evocation_context:
                                                    evocation_context.append(visual_sentence)
                                                    evocation_strength += 1
                                                    print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_strength"] = evocation_strength
                                            ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_context"] = evocation_context
            #TATE
            elif dataset_name == 'tate':
                visual_sentence = value
                for cluster_word in cluster_words:
                    for visual_word in visual_sentence:
                        if visual_word not in mismatches[cluster_name]:
                            if cluster_word in visual_word:
                                if key_URI not in ARTstract_img_dict.keys():
                                    ARTstract_img_dict[key_URI] = {}
                                ARTstract_img_dict[key_URI]["source_dataset"] = dataset_name
                                ARTstract_img_dict[key_URI]["source_id"] = key
                                if "evoked_clusters" not in ARTstract_img_dict[key_URI].keys():
                                    ARTstract_img_dict[key_URI]["evoked_clusters"] = {}
                                ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id] = {}
                                ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_name"] = cluster_name
                                ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["cluster_words"] = cluster_words
                                if key not in cluster_pic_ids:
                                    evocation_strength = 1
                                    cluster_pic_ids.append(key)
                                    evocation_context.append(visual_sentence)
                                else:
                                    if visual_sentence not in evocation_context:
                                        evocation_context.append(visual_sentence)
                                        evocation_strength += 1
                                        print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_strength"] = evocation_strength
                                ARTstract_img_dict[key_URI]["evoked_clusters"][cluster_id]["evocation_context"] = evocation_context
    with open("../output/"+dataset_name+"/ARTstract_img_dict.json", "w") as outfile:
        json.dump(ARTstract_img_dict, outfile)
    return ARTstract_img_dict

def get_cluster_dict(dataset_name, image_data, cluster_data, mismatches):
    """
    get_cluster_dict(dataset_name:str, image_data:dict cluster_data:list, mismatches:dict) -> Dict:
    get_cluster_dict() is a function that creates a dictionary of cluster data, where the keys are cluster names and the values are dictionaries containing information about the cluster's evoked images, such as the image's URI, evocation strength, and evocation context.

        Parameters:
        dataset_name: (str) name of the dataset from which the image data is sourced, such as 'artpedia' or 'advise'
        image_data: (dict) a dictionary of image data, where the keys are image IDs and the values are dictionaries of image information
        cluster_data: (list) a list of dictionaries, each containing information about a single cluster, such as its ID and name
        mismatches: (dict) a dictionary where keys are cluster names and the values are lists of words that should not be matched as evoking the cluster

        Returns:
        ARTstract_cluster_dict: (dict) a dictionary where the keys are cluster names and the values are dictionaries containing information about the cluster's evoked images, such as the image's URI, evocation strength, and evocation context.

        Outputs as CSV:
            - {dataset_name}/ARTstract_cluster_dict, containing the dictionary ARTstract_cluster_dict
    """
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
        dataset_evoker_images = dataset_name + "_evoker_images"
        ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images] = {}
        for key, value in image_data.items():
            evocation_strength = 0
            key_URI = dataset_name + '_' + key 
            evocation_words = []
            evocation_context = []
            #ARTPEDIA
            if dataset_name == 'artpedia':
                for item, item_value in value.items():
                    if item == 'visual_sentences':
                        for cluster_word in cluster_words:
                            for visual_sentence in item_value:
                                for visual_word in visual_sentence.split():
                                    if visual_word not in mismatches[cluster_name]:
                                        if cluster_word in visual_word:
                                            if key_URI not in ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images].keys():
                                                ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI] = {}
                                            if key not in cluster_pic_ids:
                                                evocation_strength = 1
                                                cluster_pic_ids.append(key)
                                                evocation_words.append(cluster_word)
                                                evocation_context.append(visual_sentence)
                                            else:
                                                if visual_sentence not in evocation_context:
                                                    evocation_words.append(cluster_word)
                                                    evocation_context.append(visual_sentence)
                                                    evocation_strength += 1
                                                    print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_word"] = cluster_word
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_strength"] = evocation_strength
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_context"] = evocation_context         
            #ADVISE
            elif dataset_name == 'advise':
                for annotation in value:
                    for item in annotation:
                        if isinstance(item, str): #if the item is a string
                            symbol_annotation = item.lower().replace('/',' ').split()
                            for cluster_word in cluster_words:
                                for symbol_word in symbol_annotation:
                                    if symbol_word not in mismatches[cluster_name]:
                                        if cluster_word in symbol_word:
                                            if key_URI not in ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images].keys():
                                                ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI] = {}
                                            if key not in cluster_pic_ids:
                                                evocation_strength = 1
                                                cluster_pic_ids.append(key)
                                                evocation_words.append(cluster_word)
                                                evocation_context.append(symbol_annotation)
                                            else:
                                                if symbol_annotation not in evocation_context:
                                                    evocation_words.append(cluster_word)
                                                    evocation_context.append(symbol_annotation)
                                                    evocation_strength += 1
                                                    print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_word"] = cluster_word
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_strength"] = evocation_strength
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_context"] = evocation_context
            #ARTEMIS
            elif dataset_name == 'artemis':
                for annotation_dict in value:
                    for item, item_value in annotation_dict.items():
                        if item == 'utterance': 
                            visual_sentence = item_value
                            for cluster_word in cluster_words:
                                for visual_word in visual_sentence.split():
                                    if visual_word not in mismatches[cluster_name]:
                                        if cluster_word in visual_word:
                                            if key_URI not in ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images].keys():
                                                ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI] = {}
                                            if key not in cluster_pic_ids:
                                                evocation_strength = 1
                                                cluster_pic_ids.append(key)
                                                evocation_words.append(cluster_word)
                                                evocation_context.append(visual_sentence)
                                            else:
                                                if visual_sentence not in evocation_context:
                                                    evocation_words.append(cluster_word)
                                                    evocation_context.append(visual_sentence)
                                                    evocation_strength += 1
                                                    print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_word"] = cluster_word
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_strength"] = evocation_strength
                                            ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_context"] = evocation_context
            #TATE
            elif dataset_name == 'tate':
                visual_sentence = value
                for cluster_word in cluster_words:
                    for visual_word in visual_sentence:
                        if visual_word not in mismatches[cluster_name]:
                            if cluster_word in visual_word:
                                if key_URI not in ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images].keys():
                                    ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI] = {}
                                if key not in cluster_pic_ids:
                                    evocation_strength = 1
                                    cluster_pic_ids.append(key)
                                    evocation_words.append(cluster_word)
                                    evocation_context.append(visual_sentence)
                                else:
                                    if visual_sentence not in evocation_context:
                                        evocation_words.append(cluster_word)
                                        evocation_context.append(visual_sentence)
                                        evocation_strength += 1
                                        print(key, 'has evocation strength', evocation_strength, 'for', cluster_name, 'for it has more than one annotation mentioning ', cluster_word)
                                ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_word"] = cluster_word
                                ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_strength"] = evocation_strength
                                ARTstract_cluster_dict[cluster_name]["evoker_images"][dataset_evoker_images][key_URI]["evocation_context"] = evocation_context
    with open("../output/"+dataset_name+"/ARTstract_cluster_dict.json", "w") as outfile:
        json.dump(ARTstract_cluster_dict, outfile)
    return ARTstract_cluster_dict













