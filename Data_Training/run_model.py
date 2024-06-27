"""
Sources used:
Datasets & DataLoaders, https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#loading-a-dataset
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
"""


import argparse
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import json
import os

from dataset_utils import FacialExpressionDataset, load_dataset
from main_model import MultiLayerFCNet
from model_variant1 import MultiLayerFCNetVariant1
from model_variant2 import MultiLayerFCNetVariant2
from model_utils import evaluate_model


def get_user_input():
    print("Choose the model to run: main, variant1, variant2, definitive, main_male, main_male2.0, main_female, main_asian, main_asian2.0, main_black, main_black2.0, main_white")
    model_choice = input("Enter your choice: ").strip()
    print("Choose the run mode: dataset or image")
    mode_choice = input("Enter your choice: ").strip()
    image_path = None
    if mode_choice == 'image':
        image_path = input("Enter the path to the image for evaluation: ").strip()
    return model_choice, mode_choice, image_path


def select_model(model_choice):
    # Import the selected model
    if model_choice == 'main':
        return MultiLayerFCNet(), 'saved_models/best_model_main.pth', 'saved_models/best_model_main_metrics.json'
    elif model_choice == 'variant1':
        return MultiLayerFCNetVariant1(), 'saved_models/best_model_variant1.pth', 'saved_models/best_model_variant1_metrics.json'
    elif model_choice == 'variant2':
        return MultiLayerFCNetVariant2(), 'saved_models/best_model_variant2.pth', 'saved_models/best_model_variant2_metrics.json'
    elif model_choice == 'definitive':
        return MultiLayerFCNet(), 'saved_models/best_model_definitive.pth', 'saved_models/best_model_definitive_metrics.json'
    elif model_choice == 'main_male':
        return MultiLayerFCNet(), 'saved_models/best_model_main_male.pth', 'saved_models/best_model_main_male_metrics.json'
    elif model_choice == 'main_male2.0':
        return MultiLayerFCNet(), 'saved_models/best_model_main_male2.0.pth', 'saved_models/best_model_main_male2.0_metrics.json'
    elif model_choice == 'main_female':
        return MultiLayerFCNet(), 'saved_models/best_model_main_female.pth', 'saved_models/best_model_main_female_metrics.json'
    elif model_choice == 'main_asian':
        return MultiLayerFCNet(), 'saved_models/best_model_main_asian.pth', 'saved_models/best_model_main_asian_metrics.json'
    elif model_choice == 'main_asian2.0':
        return MultiLayerFCNet(), 'saved_models/best_model_main_asian2.0.pth', 'saved_models/best_model_main_asian2.0_metrics.json'
    elif model_choice == 'main_black':
        return MultiLayerFCNet(), 'saved_models/best_model_main_black.pth', 'saved_models/best_model_main_black_metrics.json'
    elif model_choice == 'main_black2.0':
        return MultiLayerFCNet(), 'saved_models/best_model_main_black2.0.pth', 'saved_models/best_model_main_black2.0_metrics.json'
    elif model_choice == 'main_white':
        return MultiLayerFCNet(), 'saved_models/best_model_main_white.pth', 'saved_models/best_model_main_white_metrics.json'
    else:
        raise ValueError("Invalid model choice. Choose from 'main', 'variant1', 'variant2', 'definitive', 'main_male', 'main_male2.0', 'main_female', 'main_asian', 'main_asian2.0', 'main_black', 'main_black2.0', 'main_white'.")


def main():
    while True:
        model_choice, mode_choice, image_path = get_user_input()

        model, model_path, metrics_path = select_model(model_choice)

        # Define the transformation
        transformation = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])

        # Load and run the saved model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(torch.load(model_path))

        if mode_choice == 'dataset':
            if model_choice == 'main_male':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Male Dataset'
            elif model_choice == 'main_male2.0':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Male Dataset 2.0'
            elif model_choice == 'main_female':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Female Dataset'
            elif model_choice == 'main_asian':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Asian Dataset'
            elif model_choice == 'main_asian2.0':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Asian Dataset 2.0'
            elif model_choice == 'main_black':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Black Dataset'
            elif model_choice == 'main_black2.0':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\Black Dataset 2.0'
            elif model_choice == 'main_white':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Bias Dataset\\White Dataset'
            elif model_choice == 'definitive':
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Definitive Dataset'
            else:
                root_dir = 'C:\\Users\\mamad\\OneDrive\\Documents\\Summer 24\\COMP 472\\SmartClass-A.I.ssistant\\Dataset\\Original Dataset'

            image_paths, labels, class_names = load_dataset(root_dir)

            # Ensure the same split as in data_train.py and bias_train.py
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                image_paths, labels, test_size=0.3, random_state=42, stratify=labels)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

            test_dataset = FacialExpressionDataset(test_paths, test_labels, transform=transformation)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            print(f'Training set size: {len(train_paths)}')
            print(f'Validation set size: {len(val_paths)}')
            print(f'Test set size: {len(test_paths)}')

            # Load and print metrics from the JSON file
            try:
                with open(metrics_path, 'r') as f:
                    best_metrics = json.load(f)

                if all(key in best_metrics for key in
                       ['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']):
                    print(f"Best Model found at Epoch {best_metrics['epoch']}")
                    print(
                        f"Training Loss: {best_metrics['train_loss']:.4f}, Training Accuracy: {best_metrics['train_acc']:.2f}%, Training F1-Score: {best_metrics['train_f1']:.4f}")
                    print(
                        f"Validation Loss: {best_metrics['val_loss']:.4f}, Validation Accuracy: {best_metrics['val_acc']:.2f}%, Validation F1-Score: {best_metrics['val_f1']:.4f}")
                else:
                    print("Error: Missing keys in the metrics JSON file.")
            except FileNotFoundError:
                print(f"Error: Metrics JSON file not found at {metrics_path}.")
            except json.JSONDecodeError:
                print(f"Error: JSON decoding error in file {metrics_path}.")

            # Evaluate the model and print metrics
            metrics = evaluate_model(model, test_loader, device, class_names)

        elif mode_choice == 'image':
            while True:
                image_path = input("Enter the path to the image for evaluation (or type 'quit' to exit): ").strip()
                if image_path.lower() == 'quit':
                    break

                if not os.path.exists(image_path):
                    print("Invalid image path. Please provide a valid path to the image.")
                    continue

                image = Image.open(image_path).convert('RGB')
                image = transformation(image).unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output.data, 1)
                    predicted_class = predicted.item()

                class_names = ['Angry', 'Neutral', 'Focused', 'Happy']
                print(f"Predicted class for the input image: {class_names[predicted_class]}")

        # Ask if the user wants to evaluate another model
        continue_choice = input("Do you want to evaluate another model? (yes/no): ").strip().lower()
        if continue_choice != 'yes':
            break


if __name__ == '__main__':
    main()






