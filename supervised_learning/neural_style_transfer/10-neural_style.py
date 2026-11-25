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

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

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

    @property
    def var(self):
        """weight for the variational cost - getter"""
        return self.__var

    @var.setter
    def var(self, value):
        """weight for the variational cost- setter"""
        if not isinstance(value, (int, float)) or (value < 0):
            raise TypeError(
                "var must be a non-negative number")
        else:
            self.__var = value

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

    def generate_features(self):
        """
        Public Instance Method:
        extracts the features used to calculate neural style cost
        Sets the public instance attributes:
        gram_style_features and content_feature
        """
        style_image_255 = self.style_image * 255
        content_image_255 = self.content_image * 255
        style_input =\
            tf.keras.applications.vgg19.preprocess_input(style_image_255)
        outputs = self.model(style_input)
        self.gram_style_features = []
        for output in outputs[:-1]:
            self.gram_style_features.append(self.gram_matrix(output))

        image_input =\
            tf.keras.applications.vgg19.preprocess_input(content_image_255)
        self.content_feature = self.model(image_input)[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Public Instance Method:
        calculates the style cost for a single layer
        Returns: the layer's style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                (len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")
        c = style_output.shape[-1]
        gram_target_shape = [1, c, c]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                (gram_target.shape != gram_target_shape):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")
        else:
            return (tf.reduce_mean(tf.square(self.gram_matrix(style_output)
                                             - gram_target)))

    def style_cost(self, style_outputs):
        """
        Public Instance Method
        Calculates the style cost for generated image
        each layer should be weighted evenly with all weights summing to 1
        Returns: the style cost
        """

        if not isinstance(style_outputs, list) or \
                len(self.style_layers) != len(style_outputs):
            length = len(self.style_layers)
            raise TypeError(
                f"style_outputs must be a list with a length of {length}")
        else:
            cost = 0
            for style_output, gram_target in \
                    zip(style_outputs, self.gram_style_features):
                layer_cost = \
                    (self.layer_style_cost(style_output, gram_target)) \
                    / len(self.style_layers)
                cost += layer_cost
            return (cost)

    def content_cost(self, content_output):
        """
        Public Instance Method
        Calculates the content cost for the generated image
        Returns: the content cost
        """

        s = self.content_feature.shape

        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                content_output.shape != s:
            raise TypeError(
                f"content_output must be a tensor of shape {s}")
        else:
            return (tf.reduce_mean(tf.square(content_output
                                             - self.content_feature)))

    @staticmethod
    def variational_cost(generated_image):
        """
        static Method:
        Calculates the variational cost for the generated image
        Returns: the variational cost
        """
        return ((tf.image.total_variation(generated_image))[0])

    def total_cost(self, generated_image):
        """
        Public Instance Method:
        Calculates the total cost for the generated image

        Returns: (J, J_content, J_style, J_var)
        J is the total cost
        J_content is the content cost
        J_style is the style cost
        J_var is the variational cost
        """
        s = self.content_image.shape

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != s:
            raise TypeError(
                f"generated_image must be a tensor of shape {s}")
        else:
            generated_image_255 = generated_image * 255
            preprocessed = tf.keras.applications.vgg19.preprocess_input(
                generated_image_255)
            outputs = self.model(preprocessed)

            J_content = self.content_cost(outputs[-1])
            J_style = self.style_cost(outputs[:-1])
            J_var = self.variational_cost(generated_image)
            J = (J_content * self.alpha) + \
                (J_style * self.beta) + (J_var * self.var)
            return (J, J_content, J_style, J_var)

    def compute_grads(self, generated_image):
        """
        Public Instance Method:
        Calculates the gradients for the tf.Tensor generated image
        Returns: gradients, J_total, J_content, J_style, J_var
        J_total is the total cost for the generated image
        J_content is the content cost for the generated image
        J_style is the style cost for the generated image
        J_var is the variational cost
        """
        s = self.content_image.shape

        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != s:
            raise TypeError(
                f"generated_image must be a tensor of shape {s}")
        else:
            with tf.GradientTape() as tape:
                tape.watch(generated_image)
                J_total, J_content, J_style, J_var = self.total_cost(
                    generated_image)
            gradients = tape.gradient(J_total, generated_image)
            return (gradients, J_total, J_content, J_style, J_var)

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9,
                       beta2=0.99):
        """
        Public Instance Method:
        iterations - the number of iterations to perform gradient descent over
        step - if not None, the step at which you should print information
        about the training, including the final iteration:
        i is the iteration
        J_total is the total cost
        J_content is the content cost
        J_style is the style cost
        J_var is the variational cost
        lr - the learning rate for gradient descent
        beta1 - the beta1 parameter for gradient descent
        beta2 - the beta2 parameter for gradient descent
        Returns: generated_image, cost
        generated_image is the best generated image
        cost is the best cost
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")

        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError("step must be positive and less than iterations")

        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")

        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")

        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")

        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            gradients, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print(f"Cost at iteration {i}: {J_total}, "
                      f"content {J_content}, style {J_style}, var {J_var}")
            if i < iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

        return (best_image[0], best_cost)
