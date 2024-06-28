import os


def rename_images_in_folder(folder_path, naming_format):
    class_names = ['Angry', 'Happy', 'Neutral', 'Focused']

    for root, _, files in os.walk(folder_path):
        for i, filename in enumerate(sorted(files)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                parts = filename.split('_')
                # Check if the first part of the filename is a valid class name
                if parts[0].capitalize() in class_names:
                    expression = parts[0].capitalize()
                else:
                    # If not, determine the class name from the folder structure
                    expression = os.path.basename(root).capitalize()

                # Preserve the file extension
                file_extension = os.path.splitext(filename)[1]
                new_name = naming_format.format(expression=expression, number=i + 1) + file_extension
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


def rename_race_folders():
    base_path = ".\\Dataset\\Bias Dataset"
    race_folders = ['Asian Dataset', 'Asian Dataset 2.0', 'Black Dataset', 'Black Dataset 2.0', 'White Dataset']

    for folder in race_folders:
        folder_path = os.path.join(base_path, folder)
        if 'Asian' in folder:
            rename_images_in_folder(folder_path, "{expression}_Asian_{number:03d}")
        elif 'Black' in folder:
            rename_images_in_folder(folder_path, "{expression}_Black_{number:03d}")
        elif 'White' in folder:
            rename_images_in_folder(folder_path, "{expression}_White_{number:03d}")


def rename_gender_folders():
    base_path = ".Dataset\\Bias Dataset"
    gender_folders = ['Female Dataset', 'Male Dataset', 'Male Dataset 2.0']

    for folder in gender_folders:
        folder_path = os.path.join(base_path, folder)
        if 'Female' in folder:
            rename_images_in_folder(folder_path, "{expression}_Female_{number:03d}")
        elif 'Male' in folder:
            rename_images_in_folder(folder_path, "{expression}_Male_{number:03d}")


if __name__ == "__main__":
    rename_race_folders()
    rename_gender_folders()
