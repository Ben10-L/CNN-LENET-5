
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import load_datasets, SEED
from model import build_lenet5


# Configuration
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Optimizer selection
def get_optimizer(name, learning_rate=None):
    if name == "adam":
        lr = learning_rate if learning_rate is not None else 0.001
        return tf.keras.optimizers.Adam(learning_rate=lr)

    if name == "sgd":
        lr = learning_rate if learning_rate is not None else 0.01
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.9
        )

    raise ValueError(f"Unknown optimizer: {name}")


# Compile model
def compile_model(model, optimizer="adam", learning_rate=None):
    model.compile(
        optimizer=get_optimizer(optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Train model
def train_model(train_ds, val_ds, optimizer="adam",
                epochs=20, learning_rate=None):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_lenet5()
    model = compile_model(model, optimizer, learning_rate)

    model_path = RESULTS_DIR / f"lenet5_{optimizer}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(
            str(RESULTS_DIR / f"history_{optimizer}.csv")
        )
    ]

    print(f"Training LeNet-5 with {optimizer.upper()}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save history as JSON
    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with open(RESULTS_DIR / f"history_{optimizer}.json", "w") as f:
        json.dump(history_data, f, indent=4)

    best_epoch = int(np.argmin(history.history["val_loss"]))

    metrics = {
        "optimizer": optimizer,
        "best_epoch": best_epoch + 1,
        "val_loss": float(history.history["val_loss"][best_epoch]),
        "val_accuracy": float(history.history["val_accuracy"][best_epoch])
    }

    print("Best validation results:")
    print(f"Epoch: {metrics['best_epoch']}")
    print(f"Loss: {metrics['val_loss']:.4f}")
    print(f"Accuracy: {metrics['val_accuracy']:.4f}")

    return model, history, metrics


# Main program
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", choices=["adam", "sgd", "both"],
        default="adam"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    # Reproducibility
    tf.keras.utils.set_random_seed(SEED)

    # Load dataset
    train_ds, val_ds, test_ds, class_names = load_datasets(
        batch_size=args.batch_size
    )

    with open(RESULTS_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=4)

    optimizers = (
        ["adam", "sgd"] if args.optimizer == "both"
        else [args.optimizer]
    )

    all_results = {}

    for optimizer in optimizers:
        model, history, metrics = train_model(
            train_ds,
            val_ds,
            optimizer=optimizer,
            epochs=args.epochs,
            learning_rate=args.lr
        )

        all_results[optimizer] = metrics

    # Save validation comparison
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=4)

    best = min(
        all_results,
        key=lambda name: all_results[name]["val_loss"]
    )

    print(" Training completed ")
    print("Best optimizer:", best.upper())
    print("Best validation accuracy:",
          f"{all_results[best]['val_accuracy']:.4f}")
    print("Saved models and histories in:", RESULTS_DIR)
    print("Test dataset reserved for final evaluation.")


if __name__ == "__main__":
    main()
