#!/usr/bin/env python3
"""
Class NST that performs tasks for neural style transfer
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    defenition of class NST
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @property
    def style_image(self):
        """style_image getter"""
        return self.__style_image

    @style_image.setter
    def style_image(self, value):
        """style_image setter"""
        if not isinstance(value, np.ndarray) or (value.ndim != 3 or
                                                 value.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        else:
            self.__style_image = self.scale_image(value)

    @property
    def content_image(self):
        """content_image getter"""
        return self.__content_image

    @content_image.setter
    def content_image(self, value):
        """content_image setter"""
        if not isinstance(value, np.ndarray) or (value.ndim != 3 or
                                                 value.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        else:
            self.__content_image = self.scale_image(value)

    @property
    def alpha(self):
        """alpha - the weight for content cost - getter"""
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        """alpha - the weight for content cost - setter"""
        if not isinstance(value, (int, float)) or (value < 0):
            raise TypeError(
                "alpha must be a non-negative number")
        else:
            self.__alpha = value

    @property
    def beta(self):
        """beta - the weight for style cost - getter"""
        return self.__beta

    @beta.setter
    def beta(self, value):
        """beta - the weight for style cost - setter"""
        if not isinstance(value, (int, float)) or (value < 0):
            raise TypeError(
                "beta must be a non-negative number")
        else:
            self.__beta = value

    @property
    def model(self):
        """the Keras model used to calculate cost - getter"""
        return self.__model

    @model.setter
    def model(self, value):
        """the Keras model used to calculate cost - setter"""
        self.__model = value

    @staticmethod
    def scale_image(image):
        """
        method that rescales an image such that
        its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3)
        containing the image to be scaled

        The scaled image should be a tf.tensor
        with the shape (1, h_new, w_new, 3)
        where max(h_new, w_new) == 512
        and min(h_new, w_new) is scaled proportionately

        The image should be resized using bicubic interpolation
        Returns: the scaled image
      """
        if not isinstance(image, np.ndarray) or (image.ndim != 3 or
                                                 image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        else:
            h = image.shape[0]
            w = image.shape[1]
            maximum = max(h, w)
            if maximum == h:
                h_new = 512
                w_new = int(w / h * 512)
            else:
                w_new = 512
                h_new = int(h / w * 512)

            rescaled = tf.expand_dims(image, axis=0)
            rescaled = tf.image.resize(
                rescaled, size=(h_new, w_new), method='bicubic')
            rescaled = rescaled / 255
            rescaled = tf.clip_by_value(rescaled, 0, 1)

            return (rescaled)

    def load_model(self):
        """
        creates the model used to calculate cost
        the model should use the VGG19 Keras model as a base
        the model's input should be the same as the VGG19 input
        the model's output should be a list containing the outputs of the
        VGG19 layers listed in style_layers followed by content _layer
        saves the model in the instance attribute model
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        vgg.save("base_model.h5")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model("base_model.h5",
                                         custom_objects=custom_objects)
        vgg.trainable = False

        content_layer_list = [self.content_layer]
        layers = self.style_layers + content_layer_list
        inputs = vgg.input
        outputs = [vgg.get_layer(name).output for name in layers]

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Returns: tf.Tensor of shape (1, c, c) with gram matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                (len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a tensor of rank 4")
        else:
            batch, H, W, C = input_layer.shape
            F = tf.reshape(input_layer, (H * W, C))
            G = tf.matmul(F, F, transpose_a=True)
            G = tf.expand_dims(G, axis=0)
            return G / tf.cast(H * W, tf.float32)
