"""
sources used:
Lab Exercise #05: Artificial Neural Networks
Lab Exercise #07: Convolutional Neural Networks (CNNs)
Sklearn.metrics, https://scikit-learn.org/stable/api/sklearn.metrics.html
"""

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, data_loader, device, class_names):
    model.eval()
    test_correct = 0
    test_total = 0
    test_true = []
    test_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_true.extend(labels.cpu().numpy())
            test_pred.extend(predicted.cpu().numpy())

    test_acc = 100 * test_correct / test_total
    test_f1_weighted = f1_score(test_true, test_pred, average='weighted')

    # Calculate macro and micro averaged precision, recall, and F1-score
    precision_macro = precision_score(test_true, test_pred, average='macro')
    recall_macro = recall_score(test_true, test_pred, average='macro')
    f1_macro = f1_score(test_true, test_pred, average='macro')

    precision_micro = precision_score(test_true, test_pred, average='micro')
    recall_micro = recall_score(test_true, test_pred, average='micro')
    f1_micro = f1_score(test_true, test_pred, average='micro')

    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Weighted F1-Score: {test_f1_weighted:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}, Macro Recall: {recall_macro:.4f}, Macro F1-Score: {f1_macro:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}, Micro Recall: {recall_micro:.4f}, Micro F1-Score: {f1_micro:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(test_true, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    return {
        "accuracy": test_acc,
        "f1_weighted": test_f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro
    }
