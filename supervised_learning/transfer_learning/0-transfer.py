#!/usr/bin/env python3

from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    preprocesses data, so input data and pretrained model have alligned values
    * X / data / pictures:
        normalises / rescales our input data from
        rgb values [0, 255] to [-1, 1]
    * Y / expected outcome:
        converts expected outcome (ex 3)
        to one-hot vector (ex [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    """

    X_p = K.applications.mobilenet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


@K.utils.register_keras_serializable()
class ResizeLayer(K.layers.Layer):
    """help function"""

    def call(self, x):
        return tf.image.resize(x, (96, 96))


if __name__ == '__main__':
    """
    Python script training a convolutional neural network
    to classify the CIFAR 10 dataset:
    Keras packages a number of deep learning models
    alongside pre-trained weights into an applications module.
    """

    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    base_model = K.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(96, 96, 3),
        include_top=False
    )

    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    x = ResizeLayer()(inputs)
    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    feature_extractor = K.Model(inputs, x)
    features_train = feature_extractor.predict(X_train_p)
    features_test = feature_extractor.predict(X_test_p)

    feat_inputs = K.Input(shape=features_train.shape[1:])
    x = K.layers.Dense(256, activation='relu')(feat_inputs)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10)(x)
    head = K.Model(feat_inputs, outputs)
    head.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss=K.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ModelCheckpoint(
            'best_head.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    head.fit(
        features_train, Y_train_p,
        epochs=20,
        validation_data=(features_test, Y_test_p),
        callbacks=callbacks
    )

    final_inputs = K.Input(shape=(32, 32, 3))
    features = feature_extractor(final_inputs)
    outputs = head(features)
    model = K.Model(final_inputs, outputs)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss=K.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.save('cifar10.h5')
