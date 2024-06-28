import json
import os

def calculate_average_metrics(results):
    metrics_sums = {
        "accuracy": 0,
        "f1_weighted": 0,
        "precision_macro": 0,
        "recall_macro": 0,
        "f1_macro": 0,
        "precision_micro": 0,
        "recall_micro": 0,
        "f1_micro": 0
    }
    num_folds = len(results)

    for fold_result in results:
        test_metrics = fold_result['test_metrics']
        metrics_sums["accuracy"] += test_metrics["accuracy"]
        metrics_sums["f1_weighted"] += test_metrics["f1_weighted"]
        metrics_sums["precision_macro"] += test_metrics["precision_macro"]
        metrics_sums["recall_macro"] += test_metrics["recall_macro"]
        metrics_sums["f1_macro"] += test_metrics["f1_macro"]
        metrics_sums["precision_micro"] += test_metrics["precision_micro"]
        metrics_sums["recall_micro"] += test_metrics["recall_micro"]
        metrics_sums["f1_micro"] += test_metrics["f1_micro"]

    average_metrics = {key: value / num_folds for key, value in metrics_sums.items()}
    return average_metrics


def load_and_display_results(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)

    for fold_result in results:
        fold = fold_result['fold']
        train_metrics = fold_result['train_metrics']
        test_metrics = fold_result['test_metrics']

        print(f"Fold {fold + 1}")
        print(f"Best Model found at Epoch {train_metrics['epoch']}")
        print(
            f"Training Loss: {train_metrics['train_loss']:.4f}, Training Accuracy: {train_metrics['train_acc']:.2f}%, Training F1-Score: {train_metrics['train_f1']:.4f}")
        print(
            f"Validation Loss: {train_metrics['val_loss']:.4f}, Validation Accuracy: {train_metrics['val_acc']:.2f}%, Validation F1-Score: {train_metrics['val_f1']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test Weighted F1-Score: {test_metrics['f1_weighted']:.4f}")
        print(
            f"Macro Precision: {test_metrics['precision_macro']:.4f}, Macro Recall: {test_metrics['recall_macro']:.4f}, Macro F1-Score: {test_metrics['f1_macro']:.4f}")
        print(
            f"Micro Precision: {test_metrics['precision_micro']:.4f}, Micro Recall: {test_metrics['recall_micro']:.4f}, Micro F1-Score: {test_metrics['f1_micro']:.4f}")
        print("-" * 50)

    average_metrics = calculate_average_metrics(results)
    print("Average Metrics")
    print(f"Accuracy: {average_metrics['accuracy']:.2f}%")
    print(f"Weighted F1-Score: {average_metrics['f1_weighted']:.4f}")
    print(
        f"Macro Precision: {average_metrics['precision_macro']:.4f}, Macro Recall: {average_metrics['recall_macro']:.4f}, Macro F1-Score: {average_metrics['f1_macro']:.4f}")
    print(
        f"Micro Precision: {average_metrics['precision_micro']:.4f}, Micro Recall: {average_metrics['recall_micro']:.4f}, Micro F1-Score: {average_metrics['f1_micro']:.4f}")


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    main_model_results = os.path.join(base_path, 'saved_kfolds/main_model_k_fold_results.json')
    definitive_model_results = os.path.join(base_path, 'saved_kfolds/definitive_model_k_fold_results.json')

    print("Main Model Results")
    load_and_display_results(main_model_results)

    print("\nDefinitive Model Results")
    load_and_display_results(definitive_model_results)
