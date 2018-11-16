import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network
def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))
def drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)
def relu(x):
    return tf.nn.relu(x)
def average_pooling(x, pool_size=[2,2], stride=1, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def max_pooling(x, pool_size=[3,3], stride=1, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
def concatenation(layers) :
    return tf.concat(layers, axis=3)


### loss function define
def loss_cost(x,y):
    result=1/2*tf.reduce_mean(tf.square(x-y))
    return result
def loss_cost_l1(x,y):
    result=tf.reduce_mean(tf.abs(x-y))
    return result
def PSNR_cal(result,Y):
    erro=result-Y
    mse=tf.reduce_mean(tf.square(erro))
    psnr = 10.0*tf.log(255.0*255.0/(mse+1e-8))/tf.log(10.0)
    return psnr



