import sys
import tensorflow as tf
import numpy as np
from data_preparation import create_datasets
from residual_net import ResNet, ResidualBlock
from dense_net import DenseNet
from training import training


# Initializing the Residual Network
res_net_model = ResNet(layer_list=[
    tf.keras.layers.Conv2D(filters=3, kernel_size=5, padding='same'),
    ResidualBlock(),
    ResidualBlock(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(10)])

# Initializing the Densely Connected Convolutional Network
dense_net_model = DenseNet()



if __name__ == '__main__':
    if len(sys.argv)>1 and 'dense' in sys.argv[1].lower():
        print("Model: Densely Connected Convolutional Network")
        model = dense_net_model
    else:
        print("Model: Residual Network")
        model = res_net_model
    
    # Initialize the loss-function
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(0.1)
    # Build model to output it's summary
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    datasets = create_datasets()
    losses, accuracies = training(model, datasets,
                                  cross_entropy_loss,
                                  optimizer, epochs=10)