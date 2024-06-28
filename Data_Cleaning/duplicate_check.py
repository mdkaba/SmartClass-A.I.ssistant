"""
Sources used:
https://towardsdatascience.com/finding-duplicate-images-with-python-71c04ec8051
"""
import os
from PIL import Image
import imagehash
from collections import defaultdict
import shutil


def find_and_handle_duplicates(image_folder, duplicates_folder):
    if not os.path.exists(duplicates_folder):
        os.makedirs(duplicates_folder)

    # Dictionary to store hash values and corresponding image paths
    hash_dict = defaultdict(list)

    # Loop through all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(image_folder, filename)
            # Open image
            with Image.open(filepath) as img:
                # Calculate hash
                hash_value = imagehash.average_hash(img)
                # Store hash and image path in dictionary
                hash_dict[str(hash_value)].append(filepath)

    # Find and handle duplicates
    duplicates = [paths for paths in hash_dict.values() if len(paths) > 1]

    if duplicates:
        print("Found duplicate images:")
        for dup_group in duplicates:
            print(dup_group)
            for dup_path in dup_group[1:]:
                # Move each duplicate to the duplicates folder and delete from the original location
                shutil.move(dup_path, duplicates_folder)
    else:
        print("No duplicates found.")


image_folder = r".\Dataset\Team\Team_Cleaned\Focused"
duplicates_folder = r".\Dataset\Team\Team_Cleaned\Focused_duplicates"
find_and_handle_duplicates(image_folder, duplicates_folder)
