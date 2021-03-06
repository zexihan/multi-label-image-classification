from collections import namedtuple
from math import sqrt

import tensorflow as tf

NUM_CLASSES = 228
X_FEATURE = 'x'
LEARNING_RATE = 0.005
CUT_OFF = 0.18

def resnet_model_fn(features, labels, mode):
    """Builds a residual network."""

    # Configurations for each bottleneck group.
    BottleneckGroup = namedtuple('BottleneckGroup',
                               ['num_blocks', 'num_filters', 'bottleneck_size'])
    groups = [
        BottleneckGroup(3, 256, 64), BottleneckGroup(4, 512, 128),
        BottleneckGroup(6, 1024, 256), BottleneckGroup(3, 2048, 512)
    ]

    # groups = [
    #     BottleneckGroup(3, 128, 32), BottleneckGroup(3, 256, 64),
    #     BottleneckGroup(3, 512, 128), BottleneckGroup(3, 1024, 256)
    # ]

    x = features[X_FEATURE]
    input_shape = x.get_shape().as_list()

    # Reshape the input into the right shape if it's 2D tensor
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    training = (mode == tf.estimator.ModeKeys.TRAIN)
  
    # First convolution expands to 64 channels
    with tf.variable_scope('conv_layer1'):
        net = tf.layers.conv2d(
              x,
              filters=64,
              kernel_size=7,
              activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net, training=training)

    # Max pool
    net = tf.layers.max_pooling2d(
          net, pool_size=3, strides=2, padding='same')

    # First chain of resnets
    with tf.variable_scope('conv_layer2'):
        net = tf.layers.conv2d(
              net,
              filters=groups[0].num_filters,
              kernel_size=1,
              padding='valid')

    # Create the bottleneck groups, each of which contains `num_blocks`
    # bottleneck groups.
    for group_i, group in enumerate(groups):
        for block_i in range(group.num_blocks):
            name = 'group_%d/block_%d' % (group_i, block_i)

            # 1x1 convolution responsible for reducing dimension
            with tf.variable_scope(name + '/conv_in'):
                conv = tf.layers.conv2d(
                       net,
                       filters=group.num_filters,
                       kernel_size=1,
                       padding='valid',
                       activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv, training=training)

            with tf.variable_scope(name + '/conv_bottleneck'):
                conv = tf.layers.conv2d(
                       conv,
                       filters=group.bottleneck_size,
                       kernel_size=3,
                       padding='same',
                       activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv, training=training)

            # 1x1 convolution responsible for restoring dimension
            with tf.variable_scope(name + '/conv_out'):
                input_dim = net.get_shape()[-1].value
                conv = tf.layers.conv2d(
                       conv,
                       filters=input_dim,
                       kernel_size=1,
                       padding='valid',
                       activation=tf.nn.relu)
                conv = tf.layers.batch_normalization(conv, training=training)

            # shortcut connections that turn the network into its counterpart
            # residual function (identity shortcut)
            net = conv + net

        try:
            # upscale to the next group size
            next_group = groups[group_i + 1]
            with tf.variable_scope('block_%d/conv_upscale' % group_i):
                net = tf.layers.conv2d(
                      net,
                      filters=next_group.num_filters,
                      kernel_size=1,
                      padding='same',
                      activation=None,
                      bias_initializer=None)
        except IndexError:
            pass

    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(
          net,
          ksize=[1, net_shape[1], net_shape[2], 1],
          strides=[1, 1, 1, 1],
          padding='VALID')

    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

    # Compute logits (1 per class) and compute loss.
    logits = tf.layers.dense(net, NUM_CLASSES, activation=None)
    
    predictions = {
            'classes': tf.cast(tf.sigmoid(logits) >= CUT_OFF, tf.int8, name="class_tensor"),
            'probabilities': tf.nn.sigmoid(logits, name="prob_tensor")
    }
    
    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    # Create training op.
    if training:
        # global_step = tf.train.get_global_step()
        # learning_rate = tf.train.exponential_decay(
        # learning_rate=0.1, global_step=global_step,
        # decay_steps=100, decay_rate=0.001)
        optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    
    # Customize evaluation metric
    def meanfscore(predictions, labels):
        predictions = tf.reshape(tf.transpose(predictions), [-1])
        labels = tf.convert_to_tensor(labels)
        labels = tf.reshape(tf.transpose(labels), [-1])
        precision_micro, update_op_p = tf.metrics.precision(labels, predictions)
        recall_micro, update_op_r = tf.metrics.recall(labels, predictions)
        f1_mircro = tf.div(tf.multiply(2., tf.multiply(precision_micro, recall_micro)), tf.add(precision_micro, recall_micro), name="eval_tensor")
        return f1_mircro, tf.group(update_op_p, update_op_r)
    
    def precision_micro(predictions, labels):
        predictions = tf.reshape(tf.transpose(predictions), [-1])
        labels = tf.convert_to_tensor(labels)
        labels = tf.reshape(tf.transpose(labels), [-1])
        precision_micro, update_op_p = tf.metrics.precision(labels, predictions)
        return precision_micro, update_op_p
    
    def recall_micro(predictions, labels):
        predictions = tf.reshape(tf.transpose(predictions), [-1])
        labels = tf.convert_to_tensor(labels)
        labels = tf.reshape(tf.transpose(labels), [-1])
        recall_micro, update_op_r = tf.metrics.recall(labels, predictions)
        return recall_micro, update_op_r

    
    # Compute evaluation metrics.
    eval_metric_ops = {
        "meanfscore": meanfscore(predictions["classes"], labels),
        "precision_micro": precision_micro(predictions["classes"], labels),
        "recall_micro": recall_micro(predictions["classes"], labels)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)