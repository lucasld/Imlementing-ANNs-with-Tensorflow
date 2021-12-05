import tensorflow as tf
import numpy as np


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filter_amount) -> None:
        super(TransitionLayer, self).__init__()
        self.layer_list = [
            tf.keras.layers.Conv2D(filters=filter_amount, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        ]
    
    @tf.function
    def call(self, inputs) -> tf.Tensor:
        x = inputs
        for layer in self.layer_list:
            x = layer(x, training=True)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(self, out_filters) -> None:
        super(Block, self).__init__()
        self.block_layers = [
            tf.keras.layers.Conv2D(filters=out_filters*3,
                                   kernel_size=1,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_filters,
                                   kernel_size=1,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.ReLU()]
    
    @tf.function
    def call(self, inputs) -> tf.Tensor:
        x = inputs
        for layer in self.block_layers:
            x = layer(x, training=True)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, block_number, out_filters) -> None:
        super(DenseBlock, self).__init__()
        # Initialize as many block, transition layer pairs as specified
        # by block_number
        self.layer_blocks = [Block(out_filters) for _ in range(block_number)]
    
    @tf.function
    def call(self, inputs) -> tf.Tensor:
        x = inputs
        for block in self.layer_blocks:
            b = block(x)
            x = tf.concat([x, b], axis=-1)
        return x


class DenseNet(tf.keras.Model):
    def __init__(self, layer_list=[
        DenseBlock(block_number=3, out_filters=12),
        TransitionLayer(filter_amount=2),
        DenseBlock(block_number=3, out_filters=12),
        TransitionLayer(filter_amount=4),
        DenseBlock(block_number=3, out_filters=12),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10, 'softmax')
    ]) -> None:
        super(DenseNet, self).__init__()
        self.layer_list = layer_list
    
    @tf.function
    def call(self, inputs) -> tf.Tensor:
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x