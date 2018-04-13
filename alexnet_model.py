import tensorflow as tf

def alexnet_model_fn(features, labels, mode):
    """Model function for Alexnet."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.convert_to_tensor(features["x"])
    #print("input_layer: {}".format(input_layer.shape))

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides=4,
        padding="valid",
        activation=tf.nn.relu)
    #print("conv1: {}".format(conv1.shape))

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[3, 3], 
        strides=2, 
        padding='valid')
    #print("pool1: {}".format(pool1.shape))

    conv2 = tf.layers.conv2d(
        inputs= pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    #print("conv2: {}".format(conv2.shape))

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[3, 3], 
        strides=2, 
        padding='valid')
    #print("pool2: {}".format(pool2.shape))

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    #print("conv3: {}".format(conv3.shape))

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    #print("conv4: {}".format(conv4.shape))

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    #print("conv5: {}".format(conv5.shape))

    pool5 = tf.layers.max_pooling2d(
        inputs=conv5, 
        pool_size=[3, 3], 
        strides=2,
        padding='valid')
    #print("pool5: {}".format(pool2.shape))

    pool5_flat = tf.reshape(conv5, [-1, 12*12*256])
    #print("pool5_flat: {}".format(pool5_flat.shape))

    fc6 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    #print("dense1: {}".format(fc6.shape))  

    dropout6 = tf.layers.dropout(
        inputs=fc6, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    #print("dropout6: {}".format(dropout6.shape))

    fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu)
    #print("fc7: {}".format(fc7.shape))

    dropout7 = tf.layers.dropout(
        inputs=fc7, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    #print("dropout7: {}".format(dropout7.shape))

    # Logits Layer
    # Input Tensor Shape: [batch_size, 4096]
    # Output Tensor Shape: [batch_size, 228]
    logits = tf.layers.dense(inputs=dropout7, units=228)
    #print("logits: {}".format(logits.shape))

    # Generate Predictions
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.cast(tf.sigmoid(logits) >= 0.16, tf.int8, name="class_tensor"),
        # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="prob_tensor")
    }  

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=228)
        #w_tensor = tf.convert_to_tensor(w)
        #w_tensor = tf.reshape(w_tensor, [-1,228])
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)#, weights=w_tensor)

    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.0008)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

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
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "meanfscore": meanfscore(predictions["classes"], labels),
        "precision_micro": precision_micro(predictions["classes"], labels),
        "recall_micro": recall_micro(predictions["classes"], labels)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)