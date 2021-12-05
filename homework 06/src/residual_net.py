import tensorflow as tf
import numpy as np


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters_out) -> None:
        super(ResidualBlock, self).__init__()
        self.layer_list = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters_out, kernel_size=3, padding='same')
        ]
    
    def call(self, inputs) -> tf.Tensor:
        x = inputs
        for layer in self.layer_list:
            x = layer(x, training=True)
        # add residual block input 
        x = tf.keras.layers.Add()([x, inputs])
        return x


class ResNet(tf.keras.Model):
    """This is a custom model class
    
    :param loss_function: loss function used to calculate loss of the model
    :type loss_function: function from the tf.keras.losses module
    :param optimizer: optimizer used to apply gradients to the models
        trainable variables
    :type optimizer: function from the tf.keras.optimizers module
    :param layer_list: Contains all Layers of the model
    :type layer_list: list of CustomDense-Objects, optional
    """
    def __init__(self, layer_list) -> None:
        """Constructor function"""
        super(ResNet, self).__init__()
        self.layer_list = layer_list

    def call(self, inputs) -> tf.Tensor:
        """Compute the feed-forward pass through all layers.
        
        :param inputs: network input
        :type inputs: tf.Tensor
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x