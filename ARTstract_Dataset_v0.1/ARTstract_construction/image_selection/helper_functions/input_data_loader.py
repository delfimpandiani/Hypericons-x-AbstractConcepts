"""
get_image_data(path:str) -> Dict:
This function takes in a path as a string, opens the file in utf-8 encoding and loads the file using json.load() and returns the data in the form of dictionary

get_cluster_data(path:str) -> Dict:
This function takes in a path as a string, opens the file in utf-8 encoding, loads the file using json.load() and extracts the value of key 'data' from the dictionary and returns it.

get_mismatches(path:str) -> Dict:
This function takes in a path as a string, opens the file in utf-8 encoding and loads the file using json.load() and returns the data in the form of dictionary
"""

import json

def load_json_data(path, key=None):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        if key:
            data = data[key]
    return data

def get_image_data(path):
    return load_json_data(path)

def get_cluster_data(path):
    return load_json_data(path, key="data")

def get_mismatches(path):
    return load_json_data(path)
