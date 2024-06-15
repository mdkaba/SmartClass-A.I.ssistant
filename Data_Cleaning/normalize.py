"""
sources used:
https://www.geeksforgeeks.org/normalize-an-image-in-opencv-python/
https://neptune.ai/blog/image-processing-python-libraries-for-machine-learning
"""

import os
import cv2


def normalize_images(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Normalize and save each image
    for filename in image_files:

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping file {filename}, not a valid image.")
            continue

        # Normalize the image
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Convert the normalized image to 8-bit format before saving
        norm_img = (255 * norm_img).astype('uint8')

        # Get the output path and save the image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, norm_img)
        print(f"Normalized and saved: {output_path}")

input_folder = '.\\Dataset\\Team\\Team_Cropped'
output_folder = '.\\Dataset\\Team\\Team_Normalize'

normalize_images(input_folder, output_folder)
