
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "AMHCD_64"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
SEED = 42


# 1. Verify the dataset
def check_dataset(data_dir=DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    classes = sorted(p for p in data_dir.iterdir() if p.is_dir())

    images = [
        img
        for folder in classes
        for img in folder.rglob("*")
        if img.suffix.lower() in EXTENSIONS
    ]

    print("Number of classes:", len(classes))
    print("Number of images:", len(images))
    print("Classes:", [c.name for c in classes])

    if images:
        with Image.open(images[0]) as img:
            print("Image size:", img.size)
            print("Image mode:", img.mode)

    return classes


# 2. Load image paths and labels
def load_image_paths(data_dir=DATA_DIR):
    classes = sorted(p for p in data_dir.iterdir() if p.is_dir())
    class_names = [c.name for c in classes]

    paths, labels = [], []

    for label, folder in enumerate(classes):
        for img in sorted(folder.rglob("*")):
            if img.suffix.lower() in EXTENSIONS:
                paths.append(str(img))
                labels.append(label)

    if not paths:
        raise ValueError("No images found in the dataset.")

    return np.array(paths), np.array(labels), class_names


# 3. Preprocess images
def preprocess(path, label, image_size=IMAGE_SIZE):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape((None, None, 3))

    # Resize to 32x32
    image = tf.image.resize(image, image_size)

    # Normalize pixels to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# 4. Create TensorFlow dataset
def make_dataset(paths, labels, batch_size=32, shuffle=False,
                 image_size=IMAGE_SIZE, seed=SEED):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)

    ds = ds.map(
        lambda p, y: preprocess(p, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# 5. Split into Train / Validation / Test
def load_datasets(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                  seed=SEED):

    paths, labels, class_names = load_image_paths()

    # Train 70% / Temporary 30%
    x_train, x_temp, y_train, y_temp = train_test_split(
        paths, labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed
    )

    # Validation 15% / Test 15%
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed
    )

    train_ds = make_dataset(
        x_train, y_train, batch_size, True, image_size, seed
    )

    val_ds = make_dataset(
        x_val, y_val, batch_size, False, image_size, seed
    )

    test_ds = make_dataset(
        x_test, y_test, batch_size, False, image_size, seed
    )

    print("Dataset split:")
    print("Train:", len(x_train))
    print("Validation:", len(x_val))
    print("Test:", len(x_test))

    return train_ds, val_ds, test_ds, class_names


# 6. Test the dataset
if __name__ == "__main__":
    check_dataset()

    train_ds, val_ds, test_ds, class_names = load_datasets()

    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape)
        print("Labels shape:", labels.shape)
        print("Pixel range:", float(tf.reduce_min(images)),
              "to", float(tf.reduce_max(images)))
        print("Number of classes:", len(class_names))
