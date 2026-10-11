
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Configuration
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Plot training history
def plot_history(history, save=True):
    history = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in zip(
        axes,
        ["accuracy", "loss"],
        ["Model Accuracy", "Model Loss"]
    ):
        ax.plot(history[metric], label="Train")
        ax.plot(history[f"val_{metric}"], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(RESULTS_DIR / "training_history.png", dpi=300)

    plt.show()
    plt.close(fig)


# 2. Evaluate model on test dataset
def evaluate_model(model, test_ds, class_names, save=True):

    # Evaluate loss and compiled metrics
    results = model.evaluate(
        test_ds, verbose=0, return_dict=True
    )

    # Predict class labels
    y_true = np.concatenate([
        y.numpy() for _, y in test_ds
    ])

    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    labels = np.arange(len(class_names))

    # Classification metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    print(" Test Results ")
    for metric, value in results.items():
        print(f"{metric}: {float(value):.4f}")

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1-score: {report['macro avg']['f1-score']:.4f}")

    print(" Classification Report ")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title("Confusion Matrix - LeNet-5")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        fig.savefig(
            RESULTS_DIR / "confusion_matrix.png",
            dpi=300
        )

        metrics = {
            "test_metrics": {
                k: float(v) for k, v in results.items()
            },
            "classification_report": report
        }

        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    plt.show()
    plt.close(fig)

    return report, cm


# 3. Visualize convolutional feature maps
def plot_feature_maps(model, test_ds, max_maps=8, save=True):

    # Select convolutional layers
    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    if not conv_layers:
        print("No convolutional layers found.")
        return

    # Take one test image
    images, _ = next(iter(test_ds))
    image = images[:1]

    # Feature extraction model
    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in conv_layers]
    )

    feature_maps = feature_model.predict(image, verbose=0)

    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]

    # Display original image
    plt.figure(figsize=(3, 3))
    plt.imshow(image[0].numpy())
    plt.title("Original Image")
    plt.axis("off")

    if save:
        plt.savefig(
            RESULTS_DIR / "original_image.png",
            dpi=300
        )

    plt.show()
    plt.close()

    # Display feature maps of each convolution
    for layer, maps in zip(conv_layers, feature_maps):
        n_maps = min(max_maps, maps.shape[-1])

        fig, axes = plt.subplots(
            1, n_maps, figsize=(2 * n_maps, 3),
            squeeze=False
        )

        for i in range(n_maps):
            axes[0, i].imshow(maps[0, :, :, i], cmap="gray")
            axes[0, i].set_title(f"Filter {i + 1}")
            axes[0, i].axis("off")

        fig.suptitle(f"Feature Maps - {layer.name}")
        plt.tight_layout()

        if save:
            fig.savefig(
                RESULTS_DIR / f"feature_maps_{layer.name}.png",
                dpi=300
            )

        plt.show()
        plt.close(fig)
