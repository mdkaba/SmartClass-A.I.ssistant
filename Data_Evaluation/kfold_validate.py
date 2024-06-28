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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from torch import nn, optim

from Data_Training.dataset_utils import FacialExpressionDataset, load_dataset
from Data_Training.main_model import MultiLayerFCNet


def train_model(train_loader, val_loader, class_weights, device):
    model = MultiLayerFCNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 60
    best_val_loss = float('inf')
    patience = 6
    epochs_no_improve = 0

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
        avg_train_loss = running_loss / len(train_loader)

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
        avg_val_loss = val_loss / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    model.load_state_dict(best_model)
    return model


def evaluate_model(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)

    return precision, recall, f1, accuracy, cm


def kfold_cross_validation(dataset_path, num_folds=10):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    image_paths, labels, class_names = load_dataset(dataset_path)
    dataset = FacialExpressionDataset(image_paths, labels, transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        train_val_subset = Subset(dataset, train_val_idx)
        test_subset = Subset(dataset, test_idx)

        train_val_labels = np.array(labels)[train_val_idx]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_val_labels), y=train_val_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        train_size = int(0.85 * len(train_val_subset))
        val_size = len(train_val_subset) - train_size

        train_subset, val_subset = torch.utils.data.random_split(train_val_subset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=4)

        model = train_model(train_loader, val_loader, class_weights, device)
        precision, recall, f1, accuracy, cm = evaluate_model(model, test_loader, device)

        fold_metrics.append([precision, recall, f1, accuracy])

        print(f"Fold {fold + 1}:")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    df_metrics = pd.DataFrame(fold_metrics, columns=['Precision', 'Recall', 'F1-Score', 'Accuracy'])
    df_metrics.loc['Average'] = df_metrics.mean()
    print(df_metrics)


if __name__ == '__main__':
    dataset_paths = {
        'main_model': '.\\Dataset\\Original Dataset',
        'definitive_model': '.\\Dataset\\Definitive Dataset'
    }

    print("Choose the model to evaluate: main_model, definitive_model")
    model_choice = input("Enter your choice: ").strip().lower()

    if model_choice in dataset_paths:
        dataset_path = dataset_paths[model_choice]
        kfold_cross_validation(dataset_path)
    else:
        print("Invalid choice. Please choose from 'main_model' or 'definitive_model'.")
