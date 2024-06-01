'''
sources used:
https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
https://www.geeksforgeeks.org/numpy-concatenate-function-python/
'''

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the path to your dataset
dataset_path = "C:\\PATH\\TO\\REPO\\SmartClass-A.I.ssistant\\Dataset"

# Define the classes
classes = ['Angry', 'Neutral', 'Focused', 'Happy']

# Create a directory to save the plots
plots_dir = "C:\\PATH\\TO\\REPO\\SmartClass-A.I.ssistant\\Dataset\\Plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Function to plot class distribution and save the plot
def plot_class_distribution(class_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    plt.show()
    plt.close()

# Function to plot pixel intensity distribution for all classes in a 2x2 layout and save the plot
def plot_pixel_intensity_distribution(images_by_class, class_names):
    plt.figure(figsize=(12, 12))
    for idx, class_name in enumerate(class_names):
        plt.subplot(2, 2, idx + 1)
        images = images_by_class[class_name]
        for channel, color in zip(range(3), ['r', 'g', 'b']):
            channel_pixels = np.concatenate([img[:, :, channel].flatten() for img in images])
            plt.hist(channel_pixels, bins=50, alpha=0.7, color=color, label=f'{color.upper()} channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Pixel Intensity Distribution for {class_name}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pixel_intensity_distribution_all_classes.png'))
    plt.show()
    plt.close()

# Function to display and save sample images with histograms
def display_sample_images_with_histograms(images, class_name):
    # Randomly select 15 images
    sample_images = random.sample(images, 15)

    plt.figure(figsize=(20, 20))
    for i in range(15):
        # Display the image
        plt.subplot(5, 6, 2 * i + 1)
        plt.imshow(cv.cvtColor(sample_images[i], cv.COLOR_BGR2RGB))
        plt.title(f'{class_name} - Image {i + 1}')
        plt.axis('off')

        # Display the histogram
        plt.subplot(5, 6, 2 * i + 2)
        colors = ('r', 'g', 'b')
        for channel, color in enumerate(colors):
            histogram = cv.calcHist([sample_images[i]], [channel], None, [256], [0, 256])
            plt.plot(histogram, color=color)
            plt.xlim([0, 256])
        plt.title(f'{class_name} - Histogram {i + 1}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.axis('on')

    plt.suptitle(f'Sample Images and Histograms from {class_name} Class')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, f'sample_images_histograms_{class_name}.png'))
    plt.show()
    plt.close()

# Main function to visualize dataset and save plots
def visualize_dataset(dataset_path, classes):
    class_counts = {}
    images_by_class = {}
    for class_name in classes:
        class_folder = os.path.join(dataset_path, class_name)
        images = load_images_from_folder(class_folder)
        class_counts[class_name] = len(images)
        images_by_class[class_name] = images

        display_sample_images_with_histograms(images, class_name)

    plot_pixel_intensity_distribution(images_by_class, classes)
    plot_class_distribution(class_counts)

visualize_dataset(dataset_path, classes)
