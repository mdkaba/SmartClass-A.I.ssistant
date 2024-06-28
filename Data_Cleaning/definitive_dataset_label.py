import os
import shutil
import imagehash
from PIL import Image

# Define paths
root_dir = ".Dataset\\Bias Dataset"
definitive_dataset = os.path.join(root_dir, 'Definitive Dataset')

# Define the attributes mapping for each dataset
gender_attributes = {
    'Female Dataset': 'Female',
    'Male Dataset 2.0': 'Male'
}

race_attributes = {
    'White Dataset': 'White',
    'Black Dataset 2.0': 'Black',
    'Asian Dataset 2.0': 'Asian'
}

# Mapping class folders
class_mapping = {
    'Angry': 'Angry',
    'Focused': 'Focused',
    'Happy': 'Happy',
    'Neutral': 'Neutral'
}


# Function to hash image content using perceptual hashing
def perceptual_hash(image_path):
    with Image.open(image_path) as img:
        return imagehash.phash(img)


# Function to rename and copy images for gender
def rename_and_copy_gender_images(dataset_name, attribute):
    dataset_path = os.path.join(root_dir, dataset_name)
    for class_folder, class_name in class_mapping.items():
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.exists(class_path):
            continue
        images = os.listdir(class_path)
        for i, image_name in enumerate(images):
            src_path = os.path.join(class_path, image_name)
            new_image_name = f"{class_name}_{attribute}_{i + 1:03d}.jpg"
            dest_dir = os.path.join(definitive_dataset, class_folder)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, new_image_name)
            shutil.copy(src_path, dest_path)


# Function to rename and copy images for race
def rename_and_copy_race_images(dataset_name, attribute):
    dataset_path = os.path.join(root_dir, dataset_name)
    for class_folder, class_name in class_mapping.items():
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.exists(class_path):
            continue
        images = os.listdir(class_path)
        for i, image_name in enumerate(images):
            src_path = os.path.join(class_path, image_name)
            new_image_name = f"{class_name}_{attribute}_{i + 1:03d}.jpg"
            dest_dir = os.path.join(definitive_dataset, class_folder)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, new_image_name)
            shutil.copy(src_path, dest_path)


# Rename and copy gender images
for dataset_name, attribute in gender_attributes.items():
    rename_and_copy_gender_images(dataset_name, attribute)

# Rename and copy race images
for dataset_name, attribute in race_attributes.items():
    rename_and_copy_race_images(dataset_name, attribute)


# Combine duplicates with combined names using perceptual hashing
def combine_duplicates():
    for class_folder, class_name in class_mapping.items():
        class_path = os.path.join(definitive_dataset, class_folder)
        if not os.path.exists(class_path):
            continue
        images = os.listdir(class_path)
        image_hash_map = {}
        for image_name in images:
            src_path = os.path.join(class_path, image_name)
            image_hash = perceptual_hash(src_path)
            if image_hash in image_hash_map:
                existing_name = image_hash_map[image_hash]
                existing_parts = existing_name.split('_')[:-1]
                new_parts = image_name.split('_')[:-1]
                combined_parts = sorted(set(existing_parts + new_parts))
                combined_parts = [combined_parts[0]] + sorted(combined_parts[1:], key=lambda x: (
                'White' not in x, 'Black' not in x, 'Asian' not in x, 'Female' not in x, 'Male' not in x))
                combined_name = f"{class_name}_{'_'.join(combined_parts)}_{len(os.listdir(class_path)):03d}.jpg"
                combined_name = combined_name.replace(f"{class_name}_", "").replace(f"{class_name}_", "")
                combined_name = f"{class_name}_{combined_name}"
                combined_path = os.path.join(class_path, combined_name)
                if combined_name not in image_hash_map.values():
                    os.rename(src_path, combined_path)
                    image_hash_map[image_hash] = combined_name
                existing_path = os.path.join(class_path, existing_name)
                if os.path.exists(existing_path):
                    os.remove(existing_path)
            else:
                image_hash_map[image_hash] = image_name

        # Rename images to final format
        for i, image_name in enumerate(os.listdir(class_path)):
            src_path = os.path.join(class_path, image_name)
            parts = image_name.split('_')
            unique_parts = sorted(set(parts[1:-1]), key=lambda x: (
            'White' not in x, 'Black' not in x, 'Asian' not in x, 'Female' not in x, 'Male' not in x))
            final_name = f"{class_name}_{'_'.join(unique_parts)}_{i + 1:03d}.jpg"
            final_path = os.path.join(class_path, final_name)
            os.rename(src_path, final_path)


combine_duplicates()
