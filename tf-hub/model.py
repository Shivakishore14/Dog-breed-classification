import tensorflow as tf
import tensorflow_hub as hub
import dataset
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def adjust_image(data):
    imgs = tf.reshape(data, [-1, 299, 299, 3])
    return imgs

def model_fun(features, labels, mode):
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/classification/1", trainable=True)
    height, width = hub.get_expected_image_size(module)

    input_layer = adjust_image(features)
    outputs = module(input_layer)
    
    logits = tf.layers.dense(inputs=outputs, units=120)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        keep_checkpoint_max=1,
        log_step_count_steps=5,
        save_summary_steps=10
    )

classifier = tf.estimator.Estimator(model_fn=model_fun, model_dir="/tmp/mymodel", config=my_checkpointing_config)

# batch_x , batch_y = dataset.get_batch(size=2)
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'x': np.asarray(batch_x)},
#     y=np.asarray(batch_y),
#     batch_size=64,
#     num_epochs=None,
#     shuffle=True,
#     queue_capacity=1000,
#     num_threads=1
# )

def train_input_fn(batch_size, num_epochs):
    ds = tf.data.Dataset.from_generator(lambda : dataset.batch_yeild(size=batch_size, max_epochs=num_epochs), (tf.float32, tf.int64), (tf.TensorShape([None, 299, 299, 3]), tf.TensorShape([None])))
    return ds
classifier.train(input_fn=lambda : train_input_fn(48, 250), steps=2000)