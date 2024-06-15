"""
Sources used:
Lab Exercise #05: Artificial Neural Networks
Lab Exercise #06: Introduction to Deep Learning
Lab Exercise #07: Convolutional Neural Networks (CNNs)
Image Classification with Convolutional Neural Networks, https://www.youtube.com/watch?v=d9QHNkD_Pos
Writing Custom Datasets, DataLoaders and Transforms, https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#writing-custom-datasets-dataloaders-and-transforms
"""

import os
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader
import json

from dataset_utils import FacialExpressionDataset, load_dataset
from model_utils import evaluate_model


def get_model_choice():
    print("Choose the model to train: main, variant1, or variant2")
    model_choice = input("Enter your choice: ").strip()
    return model_choice


def select_model(model_choice):
    if model_choice == 'main':
        from main_model import MultiLayerFCNet
        model = MultiLayerFCNet()
    elif model_choice == 'variant1':
        from model_variant1 import MultiLayerFCNetVariant1
        model = MultiLayerFCNetVariant1()
    elif model_choice == 'variant2':
        from model_variant2 import MultiLayerFCNetVariant2
        model = MultiLayerFCNetVariant2()
    else:
        raise ValueError("Invalid model choice. Choose from 'main', 'variant1', or 'variant2'.")
    return model


def train_model(model_choice):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    root_dir = '.\\Dataset'
    image_paths, labels, class_names = load_dataset(root_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.3,
                                                                          random_state=42, stratify=labels)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5,
                                                                      random_state=42, stratify=temp_labels)

    train_dataset = FacialExpressionDataset(train_paths, train_labels, transform=transform)
    val_dataset = FacialExpressionDataset(val_paths, val_labels, transform=transform)
    test_dataset = FacialExpressionDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 60
    best_val_loss = float('inf')
    patience = 6
    epochs_no_improve = 0
    best_epoch = -1
    best_metrics = None
    class_names = ['Angry', 'Neutral', 'Focused', 'Happy']

    model_dir = os.path.join(os.getcwd(), 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'best_model_{model_choice}.pth')
    metrics_path = os.path.join(model_dir, f'best_model_{model_choice}_metrics.json')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_true = []
        train_pred = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_true.extend(labels.cpu().numpy())
            train_pred.extend(predicted.cpu().numpy())

        train_acc = 100 * train_correct / train_total
        train_f1 = f1_score(train_true, train_pred, average='weighted')
        avg_train_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Train F1-Score: {train_f1:.4f}")

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_true = []
        val_pred = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_true.extend(labels.cpu().numpy())
                val_pred.extend(predicted.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        val_f1 = f1_score(val_true, val_pred, average='weighted')
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Validation F1-Score: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_metrics = {
                "epoch": best_epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1
            }
            torch.save(model.state_dict(), model_path)
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    print(f"Best Model found at Epoch {best_epoch}")
    print(
        f"Training Loss: {best_metrics['train_loss']:.4f}, Training Accuracy: {best_metrics['train_acc']:.2f}%, Training F1-Score: {best_metrics['train_f1']:.4f}")
    print(
        f"Validation Loss: {best_metrics['val_loss']:.4f}, Validation Accuracy: {best_metrics['val_acc']:.2f}%, Validation F1-Score: {best_metrics['val_f1']:.4f}")

    model.load_state_dict(torch.load(model_path))
    evaluate_model(model, test_loader, device, class_names)


if __name__ == '__main__':
    model_choice = get_model_choice()
    model = select_model(model_choice)
    train_model(model_choice)
