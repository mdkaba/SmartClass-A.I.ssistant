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
import json
from torch import optim, nn
from sklearn.metrics import f1_score
from dataset_utils import FacialExpressionDataset, load_dataset
from model_utils import evaluate_model
from main_model import MultiLayerFCNet


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, class_names, fold, results, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 60
    best_val_loss = float('inf')
    patience = 6
    epochs_no_improve = 0
    best_metrics = None

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
        train_f1 = f1_score(train_true, train_pred, average='weighted', zero_division=0)
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
        val_f1 = f1_score(val_true, val_pred, average='weighted', zero_division=0)
        avg_val_loss = val_loss / len(val_loader)

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
            model_save_path = os.path.join('saved_kfolds', f'{model_name}_fold_{fold}_best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    model_load_path = os.path.join('saved_kfolds', f'{model_name}_fold_{fold}_best_model.pth')
    model.load_state_dict(torch.load(model_load_path))
    test_metrics = evaluate_model(model, test_loader, device, class_names)
    results.append({"fold": fold, "train_metrics": best_metrics, "test_metrics": test_metrics})


def k_fold_cross_validation(model_name, dataset_path, n_splits=10):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    image_paths, labels, class_names = load_dataset(dataset_path)
    dataset = FacialExpressionDataset(image_paths, labels, transform=transform)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []

    os.makedirs('saved_kfolds', exist_ok=True)

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        train_val_subset = Subset(dataset, train_val_idx)
        test_subset = Subset(dataset, test_idx)

        train_val_loader = DataLoader(train_val_subset, batch_size=64, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=4)

        # Further split train_val_loader into train_loader and val_loader
        train_size = int(0.85 * len(train_val_subset))
        val_size = len(train_val_subset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_val_subset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

        model = MultiLayerFCNet().to(device)
        train_and_evaluate(model, train_loader, val_loader, test_loader, device, class_names, fold, results, model_name)

    # Save results to a JSON file
    results_save_path = os.path.join('saved_kfolds', f'{model_name}_k_fold_results.json')
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    original_dataset_path = '.\\Dataset\\Original Dataset'
    definitive_dataset_path = '.\\Dataset\\Definitive Dataset'

    k_fold_cross_validation('main_model', original_dataset_path)
    k_fold_cross_validation('definitive_model', definitive_dataset_path)

