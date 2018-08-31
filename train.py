import dataset
import numpy as np
import tensorflow as tf
import os


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 128, 128, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=dataset.n_classes)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    acc = tf.metrics.accuracy(
          labels=labels, predictions=predictions['classes'])
    return train_op, loss, logits, acc

x = tf.placeholder(tf.float32, shape=[None, 128*128])
y = tf.placeholder(tf.int32, shape=[None])
global_step_op = tf.train.create_global_step()

train_op, loss, logits, accuracy = cnn_model_fn({'x': x}, y, None)


out_dir = os.path.abspath("logs/")
checkpoint_prefix = 'checkpoints/'
print("Writing to {}\n".format(out_dir))
loss_summary = tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge([loss_summary])

saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)


with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
    try:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_prefix)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
    except:
        print "cannot load checkpoints"

    sess.run(tf.global_variables_initializer())
    i = 0
    while dataset.n_epochs <= 1000:
        i += 1
        batch_x, batch_y = dataset.get_batch(size=64)
    
        _, loss_, logits_, global_step, summary_ = sess.run([train_op, loss, logits, tf.train.get_global_step(), summary_op],
                                                  feed_dict={x: batch_x, y: batch_y})
        summary_writer.add_summary(summary_, global_step)
#         global_step = tf.train.get_global_step()
        if i % 2 == 0:
            print "step : {}, epoch : {} , loss : {} ".format(global_step, dataset.n_epochs, loss_)
            saver.save(sess, checkpoint_prefix, global_step=global_step)
