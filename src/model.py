
import tensorflow as tf
from tensorflow.keras import layers, Model


# Convolution layer
def conv_layer(x, filters, kernel_size=5, activation="tanh", name=None):
    return layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="valid",
        activation=activation,
        name=name
    )(x)



# Average pooling layer
def pool_layer(x, pool_size=2, strides=2, name=None):
    return layers.AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        name=name
    )(x)


# Fully connected layer
def dense_layer(x, units, activation="tanh", name=None):
    return layers.Dense(
        units=units,
        activation=activation,
        name=name
    )(x)


# LeNet-5 architecture
def build_lenet5(
    input_shape=(32, 32, 3),
    num_classes=33,
    conv_filters=(6, 16),
    dense_units=(120, 84),
    kernel_size=5,
    pool_size=2,
    activation="tanh"
):

    inputs = layers.Input(shape=input_shape, name="Input_RGB")

    # C1: 32x32x3 -> 28x28x6
    x = conv_layer(
        inputs, conv_filters[0],
        kernel_size, activation, name="C1"
    )

    # S2: 28x28x6 -> 14x14x6
    x = pool_layer(x, pool_size, name="S2")

    # C3: 14x14x6 -> 10x10x16
    x = conv_layer(
        x, conv_filters[1],
        kernel_size, activation, name="C3"
    )

    # S4: 10x10x16 -> 5x5x16
    x = pool_layer(x, pool_size, name="S4")

    # Flatten: 5x5x16 -> 400
    x = layers.Flatten(name="Flatten")(x)

    # C5: 400 -> 120
    x = dense_layer(
        x, dense_units[0], activation, name="C5"
    )

    # F6: 120 -> 84
    x = dense_layer(
        x, dense_units[1], activation, name="F6"
    )

    # Output: 84 -> 33 classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="Output"
    )(x)

    return Model(inputs, outputs, name="LeNet5_RGB")


# Test model
if __name__ == "__main__":

    model = build_lenet5()

    model.summary()

    # Verify output shape
    sample = tf.random.uniform((1, 32, 32, 3))
    prediction = model(sample)

    print("\nInput shape:", sample.shape)
    print("Output shape:", prediction.shape)
    print("Total parameters:", model.count_params())
