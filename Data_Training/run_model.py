"""
Sources used:
Datasets & DataLoaders, https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#loading-a-dataset
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
"""

import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_utils import FacialExpressionDataset, load_dataset
from main_model import MultiLayerFCNet
from model_variant1 import MultiLayerFCNetVariant1
from model_variant2 import MultiLayerFCNetVariant2
from model_utils import evaluate_model
import os
from PIL import Image
from sklearn.model_selection import train_test_split


def get_user_choices():
    # Prompt user for model choice
    print("Choose the model to run: main, variant1, or variant2")
    model_choice = input("Enter your choice: ").strip()

    # Prompt user for mode choice
    print("Choose the run mode: dataset or image")
    mode_choice = input("Enter your choice: ").strip()

    return model_choice, mode_choice


def select_model(model_choice):
    # Import the selected model
    if model_choice == 'main':
        return MultiLayerFCNet()
    elif model_choice == 'variant1':
        return MultiLayerFCNetVariant1()
    elif model_choice == 'variant2':
        return MultiLayerFCNetVariant2()
    else:
        raise ValueError("Invalid model choice. Choose from 'main', 'variant1', or 'variant2'.")


def run_model(model, model_choice, mode_choice):
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'saved_models')
    model_path = os.path.join(model_dir, f'best_model_{model_choice}.pth')
    metrics_path = os.path.join(model_dir, f'best_model_{model_choice}_metrics.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")

    model.load_state_dict(torch.load(model_path))

    if mode_choice == 'dataset':
        root_dir = '.\\Dataset'
        image_paths, labels, class_names = load_dataset(root_dir)

        # Ensure the same split as in data_train.py
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=0.3, random_state=42, stratify=labels)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

        test_dataset = FacialExpressionDataset(test_paths, test_labels, transform=transformation)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        # Load best model metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                best_metrics = json.load(f)
            print(f"Best Model found at Epoch {best_metrics['epoch']}")
            print(
                f"Training Loss: {best_metrics['train_loss']:.4f}, Training Accuracy: {best_metrics['train_acc']:.2f}%, Training F1-Score: {best_metrics['train_f1']:.4f}")
            print(
                f"Validation Loss: {best_metrics['val_loss']:.4f}, Validation Accuracy: {best_metrics['val_acc']:.2f}%, Validation F1-Score: {best_metrics['val_f1']:.4f}")

        evaluate_model(model, test_loader, device, class_names)

    elif mode_choice == 'image':
        image_path = input("Enter the path to the image for evaluation: ").strip().strip('"')
        if not os.path.exists(image_path):
            raise ValueError("Invalid image path. Please provide a valid path to the image.")
        image = Image.open(image_path).convert('RGB')
        image = transformation(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predicted_class = predicted.item()

        class_names = ['Angry', 'Neutral', 'Focused', 'Happy']
        print(f"Predicted class for the input image: {class_names[predicted_class]}")


if __name__ == '__main__':
    model_choice, mode_choice = get_user_choices()
    model = select_model(model_choice)
    run_model(model, model_choice, mode_choice)
