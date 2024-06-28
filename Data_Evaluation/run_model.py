"""
Sources used:
Datasets & DataLoaders, https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#loading-a-dataset
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
"""

import argparse
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import json

from Data_Training.main_model import MultiLayerFCNet
from Data_Training.model_variant1 import MultiLayerFCNetVariant1
from Data_Training.model_variant2 import MultiLayerFCNetVariant2
from Data_Training.dataset_utils import FacialExpressionDataset, load_dataset
from Data_Training.model_utils import evaluate_model

def get_user_input():
    print("Choose the model to run: main, definitive, variant1, variant2, main_male, main_female, main_asian, main_black, main_white, main_male2.0, main_asian2.0, main_black2.0")
    model_choice = input("Enter your choice: ").strip()
    print("Choose the run mode: dataset or image")
    mode_choice = input("Enter your choice: ").strip()
    image_path = None
    if mode_choice == 'image':
        image_path = input("Enter the path to the image for evaluation: ").strip()
    return model_choice, mode_choice, image_path

def main():
    while True:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, help='Choose the model to run: main, variant1, variant2, main_male, main_female, main_asian, main_black, main_white, main_male2, main_asian2, main_black2, definitive')
        parser.add_argument('--mode', type=str, choices=['dataset', 'image'], help='Run mode: dataset or image')
        parser.add_argument('--image_path', type=str, help='Path to the image for evaluation (required if mode is "image")')
        args = parser.parse_args()

        if not args.model or not args.mode:
            args.model, args.mode, args.image_path = get_user_input()

        if args.model == 'main':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_metrics.json'
        elif args.model == 'variant1':
            model = MultiLayerFCNetVariant1()
            model_path = 'Data_Evaluation/saved_models/best_model_variant1.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_variant1_metrics.json'
        elif args.model == 'variant2':
            model = MultiLayerFCNetVariant2()
            model_path = 'Data_Evaluation/saved_models/best_model_variant2.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_variant2_metrics.json'
        elif args.model == 'main_male':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_male.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_male_metrics.json'
        elif args.model == 'main_female':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_female.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_female_metrics.json'
        elif args.model == 'main_asian':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_asian.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_asian_metrics.json'
        elif args.model == 'main_black':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_black.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_black_metrics.json'
        elif args.model == 'main_white':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_white.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_white_metrics.json'
        elif args.model == 'main_male2.0':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_male2.0.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_male2.0_metrics.json'
        elif args.model == 'main_asian2.0':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_asian2.0.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_asian2.0_metrics.json'
        elif args.model == 'main_black2.0':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_main_black2.0.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_main_black2.0_metrics.json'
        elif args.model == 'definitive':
            model = MultiLayerFCNet()
            model_path = 'Data_Evaluation/saved_models/best_model_definitive.pth'
            metrics_path = 'Data_Evaluation/saved_models/best_model_definitive_metrics.json'
        else:
            raise ValueError("Invalid model choice. Choose from 'main', 'variant1', 'variant2', 'main_male', 'main_female', 'main_asian', 'main_black', 'main_white', 'main_male2', 'main_asian2', 'main_black2', 'definitive'.")

        print(f"Loading model from {model_path}")  # Debug print
        transformation = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            break

        model.load_state_dict(torch.load(model_path))

        if args.mode == 'dataset':
            if args.model == 'main_male':
                root_dir = 'Dataset/Bias Datasets/Male Dataset'
            elif args.model == 'main_male2.0':
                root_dir = 'Dataset/Bias Datasets/Male Dataset 2.0'
            elif args.model == 'main_female':
                root_dir = 'Dataset/Bias Datasets/Female Dataset'
            elif args.model == 'main_asian':
                root_dir = 'Dataset/Bias Datasets/Asian Dataset'
            elif args.model == 'main_asian2.0':
                root_dir = 'Dataset/Bias Datasets/Asian Dataset 2.0'
            elif args.model == 'main_black':
                root_dir = 'Dataset/Bias Datasets/Black Dataset'
            elif args.model == 'main_black2.0':
                root_dir = 'Dataset/Bias Datasets/Black Dataset 2.0'
            elif args.model == 'main_white':
                root_dir = 'Dataset/Bias Datasets/White Dataset'
            elif args.model == 'definitive':
                root_dir = 'Dataset/Definitive Dataset'
            else:
                root_dir = 'Dataset/Original Dataset'

            image_paths, labels, class_names = load_dataset(root_dir)

            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                image_paths, labels, test_size=0.3, random_state=42, stratify=labels)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

            test_dataset = FacialExpressionDataset(test_paths, test_labels, transform=transformation)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

            print(f'Training set size: {len(train_paths)}')
            print(f'Validation set size: {len(val_paths)}')
            print(f'Test set size: {len(test_paths)}')

            try:
                with open(metrics_path, 'r') as f:
                    best_metrics = json.load(f)

                if all(key in best_metrics for key in ['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1']):
                    print(f"Best Model found at Epoch {best_metrics['epoch']}")
                    print(f"Training Loss: {best_metrics['train_loss']:.4f}, Training Accuracy: {best_metrics['train_acc']:.2f}%, Training F1-Score: {best_metrics['train_f1']:.4f}")
                    print(f"Validation Loss: {best_metrics['val_loss']:.4f}, Validation Accuracy: {best_metrics['val_acc']:.2f}%, Validation F1-Score: {best_metrics['val_f1']:.4f}")
                else:
                    print("Error: Missing keys in the metrics JSON file.")
            except FileNotFoundError:
                print(f"Error: Metrics JSON file not found at {metrics_path}.")
            except json.JSONDecodeError:
                print(f"Error: JSON decoding error in file {metrics_path}.")

            metrics = evaluate_model(model, test_loader, device, class_names)

        elif args.mode == 'image':
            while True:
                if not args.image_path:
                    args.image_path = input("Enter the path to the image for evaluation (or type 'quit' to exit): ").strip()
                    if args.image_path.lower() == 'quit':
                        break

                args.image_path = args.image_path.strip('"')  # Remove surrounding quotes if any

                if not os.path.exists(args.image_path):
                    print("Invalid image path. Please provide a valid path to the image.")
                    args.image_path = None
                    continue

                image = Image.open(args.image_path).convert('RGB')
                image = transformation(image).unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output.data, 1)
                    predicted_class = predicted.item()

                class_names = ['Angry', 'Neutral', 'Focused', 'Happy']
                print(f"Predicted class for the input image: {class_names[predicted_class]}")
                args.image_path = None

        again = input("Do you want to evaluate another model or image? (yes/no): ").strip().lower()
        if again != 'yes':
            break

if __name__ == '__main__':
    main()

