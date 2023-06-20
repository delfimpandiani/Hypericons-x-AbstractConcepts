"""
Script: create_tate_dataset.py
Description: This script is used to download images from the Tate art collection dataset and save them to a local directory. It reads the information from a CSV file containing artwork data and uses the URLs provided to download the corresponding images. It also handles cases where the original URL is not accessible by generating alternative URLs based on a regular expression pattern.

Dependencies:
- os
- csv
- urllib.request
- re

Input:
- tate_artwork_data.csv: A CSV file containing artwork data. The file should have two columns:
    1. Acno: The unique identification number for each artwork.
    2. URL_Suffix: The URL suffix for each artwork's image.

Output:
- Downloaded images: The script saves the downloaded images to the "../input_data/tate_dataset/" directory.

Usage: Run the script to download the images from the Tate art collection dataset and save them to the local directory.

"""
import os
import csv
import urllib.request
import re

def create_tate_dataset():
    dataset_name = "tate"
    base_url = "https://media.tate.org.uk/art/images/work/"
    if not os.path.exists("../Local_input_data/tate_dataset/"):
        os.mkdir("../Local_input_data/tate_dataset/")
    destination_path = "../Local_input_data/tate_dataset/"
    failed = []
    succeed = []
    with open("../Local_input_data/tate_thumbnail_urls.csv.csv") as csvfile:
        reader = csv.reader(csvfile) 
        for i in range(1): # Skip the first row
            next(reader) 
        # Iterate through the remaining rows 
        for row in reader:
            acno = row[0] 
            url_suffix = row[1] 
            url = base_url + url_suffix
            regex = re.sub(r'https://media.tate.org.uk/art/images/work/(.*?)_.*', r'https://media.tate.org.uk/art/images/work/\1_', url)
            print(regex)
            if url == "https://media.tate.org.uk/art/images/work/":
                continue
            else:
                new_name = acno+".jpg"
                new_path = os.path.join(destination_path, new_name)
                try:
                    # Download the image and save it to the "tate_dataset" directory 
                    urllib.request.urlretrieve(url, new_path)
                    succeed.append(url)
                except:
                    try:
                        for i in range(100):
                            path = regex + str(i) + ".jpg"
                            new_name = acno+".jpg"
                            new_path = os.path.join(destination_path, new_name)
                            try:
                                # Download the image and save it to the "tate_dataset" directory 
                                urllib.request.urlretrieve(path, new_path)
                                succeed.append(url)
                            except:
                                continue
                    except:
                        print("could not find", url)
                        failed.append(url)
                        continue
            # else:
            #     try:
            #         for i in range(100):
            #             path = regex + str(i) + ".jpg"
            #             new_name = acno+".jpg"
            #             new_path = os.path.join(destination_path, new_name)
            #             try:
            #                 # Download the image and save it to the "tate_dataset" directory 
            #                 urllib.request.urlretrieve(path, new_path)
            #                 succeed.append(url)
            #             except:
            #                 continue
            #     except:
            #         print("could not find", url)
            #         failed.append(url)
            #         continue

    # print("For dataset", dataset_name, "SUCCEEDED", succeed)
    print("For dataset", dataset_name, "NUMBER SUCCEEDED", len(succeed))
    # print("For dataset", dataset_name, "FAILED", failed)
    print("For dataset", dataset_name, "NUMBER FAILED", len(failed))
    return 

create_tate_dataset()