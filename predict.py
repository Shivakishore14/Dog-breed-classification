import dataset
import numpy as np
import tensorflow as tf
import model
import os

x = tf.placeholder(tf.float32, shape=[None, 128*128])

checkpoint_prefix = 'checkpoints/'
graph = tf.get_default_graph()

with tf.Session(graph=graph) as sess:

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_prefix)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    

    x = graph.get_tensor_by_name("Placeholder_2:0")
    logits = graph.get_tensor_by_name("dense_1/BiasAdd:0")
    batch_x, batch_y = dataset.get_batch(size=64)
    logits_ = sess.run(logits, feed_dict={x: batch_x})
    count = 0
    for logit_, y_ in zip(logits_, batch_y):
        # print np.argmax(logit_), y_
        if np.argmax(logit_) == y_:
            count += 1
    print count
    