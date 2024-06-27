"""
Source used:
https://learnopencv.com/image-resizing-with-opencv/
"""

'''import cv2 as cv
import os


# Function to resize images
def resize_images(input_folder, output_folder, size=(96, 96)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    for file in files:
        input_file_path = os.path.join(input_folder, file)
        if os.path.isfile(input_file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv.imread(input_file_path)

            if image is not None:
                # Resize the image
                resized_image = cv.resize(image, size)
                output_file_path = os.path.join(output_folder, file)

                # Save the resized image
                cv.imwrite(output_file_path, resized_image)
                print(f"Resized and saved {file} to {output_file_path}")
            else:
                print(f"Failed to read {file}")


input_folder = "C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\new_image"
output_folder = "C:\\Users\\mamad\OneDrive\\Documents\\Summer 24\\COMP 472\\new_image\\resized_images"

# Call the function to resize images
resize_images(input_folder, output_folder, size=(96, 96))
'''
"""
Source used:
https://learnopencv.com/image-resizing-with-opencv/
"""

import cv2 as cv
import os


# Function to resize images
def resize_images(input_folder, size=(96, 96)):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                output_file_subfolder = os.path.join(root, f"{folder_name}_resized")

                if not os.path.exists(output_file_subfolder):
                    os.makedirs(output_file_subfolder)

                if os.path.isfile(input_file_path):
                    image = cv.imread(input_file_path)

                    if image is not None:
                        # Resize the image
                        resized_image = cv.resize(image, size)
                        output_file_path = os.path.join(output_file_subfolder, file)

                        # Save the resized image
                        cv.imwrite(output_file_path, resized_image)
                        print(f"Resized and saved {file} to {output_file_path}")
                    else:
                        print(f"Failed to read {file}")


input_folder = "C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\a"

# Call the function to resize images
resize_images(input_folder, size=(96, 96))
