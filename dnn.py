import tensorflow as tf

HEIGHT = 28
WIDTH = 28
NCLASSES = 10


def dnn_model(img, mode, hparams):
    # returns a tensor with shape as "shape" param
    X = tf.reshape(tensor=img, shape=[-1, HEIGHT*WIDTH])

    # first layer
    # This layer implements the operation: outputs = activation(inputs * kernel + bias) where activation
    # is the activation function passed as the activation argument (if not None), kernel is a weights
    # matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is True).
    h1 = tf.layers.dense(inputs=X, units=300, activation=tf.nn.relu)
    h2 = tf.layers.dense(inputs=h1, units=100, activation=tf.nn.relu)
    h3 = tf.layers.dense(inputs=h2, units=50, activation=tf.nn.relu)
    ylogits = tf.layers.dense(inputs=h3, units=NCLASSES, activation=None)
    return ylogits, NCLASSES
