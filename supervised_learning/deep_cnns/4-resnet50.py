#!/usr/bin/env python3

"""ResNet-50"""

from tensorflow import keras as K


def resnet50():
    """
    function that builds the ResNet-50 architecture
    as described in Deep Residual Learning for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)

    All convolutions inside and outside the blocks should be followed by
    batch normalization along the channels axis
    and a rectified linear activation (ReLU), respectively.

    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero

    Returns: the keras model
    """

    identity_block = __import__('2-identity_block').identity_block
    projection_block = __import__('3-projection_block').projection_block

    inputs = K.Input(shape=(224, 224, 3))

    # conv1 : 7×7, 64, stride 2 → BN → ReLU
    A = K.layers.Conv2D(
        64, (7, 7), strides=2, padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0))(inputs)
    A = K.layers.BatchNormalization(axis=3)(A)
    A = K.layers.Activation('relu')(A)

    # pool : 3×3, stride 2
    A = K.layers.MaxPooling2D((3, 3), strides=2, padding='same')(A)

    # conv2_x : projection(s=1) + 2× identity   filters=(64, 64, 256)
    A = projection_block(A, (64, 64, 256), 1)
    A = identity_block(A, (64, 64, 256))
    A = identity_block(A, (64, 64, 256))

    # conv3_x : projection(s=2) + 3× identity   filters=(128, 128, 512)
    A = projection_block(A, (128, 128, 512), 2)
    A = identity_block(A, (128, 128, 512))
    A = identity_block(A, (128, 128, 512))
    A = identity_block(A, (128, 128, 512))

    # conv4_x : projection(s=2) + 5× identity   filters=(256, 256, 1024)
    A = projection_block(A, (256, 256, 1024), 2)
    A = identity_block(A, (256, 256, 1024))
    A = identity_block(A, (256, 256, 1024))
    A = identity_block(A, (256, 256, 1024))
    A = identity_block(A, (256, 256, 1024))
    A = identity_block(A, (256, 256, 1024))

    # conv5_x : projection(s=2) + 2× identity   filters=(512, 512, 2048)
    A = projection_block(A, (512, 512, 2048), 2)
    A = identity_block(A, (512, 512, 2048))
    A = identity_block(A, (512, 512, 2048))

    # Global Average Pooling
    A = K.layers.GlobalAveragePooling2D()(A)

    # Dense(1000, softmax)
    outputs = K.layers.Dense(1000, activation='softmax')(A)

    return K.Model(inputs, outputs)
