import dataset
import numpy as np
import tensorflow as tf
import model
import os
import datetime

x = tf.placeholder(tf.float32, shape=[None, 128*128*3], name='x')
y = tf.placeholder(tf.int32, shape=[None], name='y')
global_step_op = tf.train.create_global_step()

train_op, loss, logits, accuracy = model.cnn_model_fn({'x': x}, y, None)


out_dir = os.path.abspath("logs/")
checkpoint_prefix = 'checkpoints/'
print("Writing to {}\n".format(out_dir))
loss_summary = tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge([loss_summary])

saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    while dataset.n_epochs <= 250:
        start_time = datetime.datetime.now()
        batch_x, batch_y = dataset.get_batch(size=128)
    
        _, loss_, logits_, global_step, summary_ = sess.run([train_op, loss, logits, tf.train.get_global_step(), summary_op],
                                                  feed_dict={x: batch_x, y: batch_y})
        summary_writer.add_summary(summary_, global_step)
        if global_step % 2 == 0:
            saver.save(sess, checkpoint_prefix, global_step=global_step)
            end_time = datetime.datetime.now()
            print "step : {}, epoch : {} , loss : {} , time: {}s".format(global_step, dataset.n_epochs, loss_, (end_time - start_time).seconds)
