import torch
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device, class_names):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(
        targets,
        predictions
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            targets,
            predictions,
            target_names=class_names
        )
    )

    cm = confusion_matrix(
        targets,
        predictions
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Real")

    plt.show()

    return accuracy
