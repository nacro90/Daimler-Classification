import sys
import time
import tensorflow as tf

import datasource as ds
from checkpoint_manager import CheckpointManager

IMAGE_WIDTH = 18
IMAGE_HEIGHT = 36
IMAGE_CHANNEL = 1

N_CLASSES = 1

EPOCH_LENGTH = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.001

RANDOM_SEED = 1

WEIGHT_COUNTER = 0
BIAS_COUNTER = 0
CONVOLUTION_COUNTER = 0
POOLING_COUNTER = 0

def main():

    tf.reset_default_graph()

    TEST = None
    NETWORK_NUMBER = None

    if len(sys.argv) > 2 and sys.argv[1] is not None and sys.argv[1] is not None:
        if sys.argv[1] == 'test':
            TEST = True
        elif sys.argv[1] == 'train':
            TEST = False
        else:
            raise ValueError("Invalid command line argument")

        NETWORK_NUMBER = int(sys.argv[2])
    else:
        raise ValueError("Enter a command line argument [test/train]")

    print(NETWORK_NUMBER)

    input_placeholder = tf.placeholder(
        tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name='input_placeholder')

    output_placeholder = tf.placeholder(
        tf.float32, shape=[None, N_CLASSES], name='output_placeholder')

    layer_conv_1, weights_conv_1 = new_conv_layer(
        input=input_placeholder,
        num_input_channels=IMAGE_CHANNEL,
        filter_size=5,
        num_filters=64,
        pooling=2
    )

    layer_conv_2, weights_conv_2 = new_conv_layer(
        input=layer_conv_1,
        num_input_channels=64,
        filter_size=3,
        num_filters=128,
        pooling=2
    )

    layer_conv_3, weights_conv_3 = new_conv_layer(
        input=layer_conv_2,
        num_input_channels=128,
        filter_size=3,
        num_filters=128,
        pooling=None
    )

    layer_conv_4, weights_conv_4 = new_conv_layer(
        input=layer_conv_3,
        num_input_channels=128,
        filter_size=3,
        num_filters=128,
        pooling=None
    )

    layer_conv_5, weights_conv_5 = new_conv_layer(
        input=layer_conv_4,
        num_input_channels=128,
        filter_size=3,
        num_filters=256,
        pooling=3
    )

    layer_flat, num_features = flatten_layer(layer_conv_5)

    layer_fc_1 = new_fc_layer(
        input=layer_flat, num_inputs=num_features, num_outputs=2048)

    layer_fc_1 = tf.nn.sigmoid(layer_fc_1)

    if TEST is not True:
        # pass
        layer_fc_1 = tf.nn.dropout(layer_fc_1, 0.6)

    layer_fc_2 = new_fc_layer(
        input=layer_fc_1, num_inputs=2048, num_outputs=2048)

    layer_fc_2 = tf.nn.sigmoid(layer_fc_2)

    if TEST is not True:
        # pass
        layer_fc_2 = tf.nn.dropout(layer_fc_2, 0.6)

    layer_output = new_fc_layer(
        input=layer_fc_2, num_inputs=2048, num_outputs=N_CLASSES)

    layer_output = tf.nn.sigmoid(layer_output)

    cost = tf.reduce_sum(tf.squared_difference(layer_output, output_placeholder) / 2)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)


    predictions = tf.round(layer_output)
    prediction_equalities = tf.equal(predictions, output_placeholder)
    accuracy = tf.reduce_mean(tf.cast(prediction_equalities, tf.float32))

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)

        checkpoint_manager = CheckpointManager(NETWORK_NUMBER)
        if TEST is False:
            train_nn(sess, NETWORK_NUMBER, input_placeholder,
                     output_placeholder, accuracy, cost, optimizer)
            checkpoint_manager.save_model(sess)
        elif TEST is True:
            checkpoint_manager.restore_model(sess)
            test_nn(sess, NETWORK_NUMBER, input_placeholder,
                    output_placeholder, accuracy, cost)
            test_nn(sess, NETWORK_NUMBER, input_placeholder,
                    output_placeholder, accuracy, cost)
        else:
            raise ValueError("Invalid TEST value!")


def train_nn(sess, number, input_placeholder, output_placeholder, accuracy, cost, optimizer):
    global TEST

    checkpoint_manager = CheckpointManager(number)

    checkpoint_manager.on_training_start(
        ds.DATASET_FOLDER, EPOCH_LENGTH, BATCH_SIZE,
        LEARNING_RATE, "AdamOptimizer", True)

    for batch_index, batch_images, batch_labels in ds.training_batch_generator(BATCH_SIZE):

        print("\nStarting batch {}".format(batch_index + 1))

        for current_epoch in range(EPOCH_LENGTH):

            feed = {
                input_placeholder: batch_images,
                output_placeholder: batch_labels
            }

            epoch_accuracy, epoch_cost, _ = sess.run(
                [accuracy, cost, optimizer], feed_dict=feed)
            print("Batch {:3}, Epoch {:3} -> Accuracy: {:3.1%}, Cost: {}".format(
                batch_index + 1, current_epoch + 1, epoch_accuracy, epoch_cost))

            checkpoint_manager.on_epoch_completed()

        TEST = True

        batch_accuracy_training, batch_cost_training = sess.run(
            [accuracy, cost], feed_dict=feed)

        print("Training batch {} has been finished. Accuracy: {:3.1%}, Cost: {}".format(
            batch_index + 1, batch_accuracy_training, batch_cost_training))

        batch_accuracy_test, batch_cost_test = \
            test_nn(sess, number, input_placeholder,
                    output_placeholder, accuracy, cost)

        TEST = False

        checkpoint_manager.on_batch_completed(
            batch_cost_training, batch_accuracy_training, batch_accuracy_test)

        checkpoint_manager.save_model(sess)

    print("\nTraining finished at {}!".format(time.asctime()))

    overall_accuracy, overall_cost = \
        test_nn(number, input_placeholder, output_placeholder, accuracy, cost)
    
    checkpoint_manager.on_training_completed("{:3.1%}".format(overall_accuracy))


def test_nn(sess, number, input_placeholder, output_placeholder, accuracy, cost):

    total_accuracy = 0
    total_cost = 0
    batches = None
    for batch_index, test_images, test_labels in ds.test_batch_generator(BATCH_SIZE):

        feed = {
            input_placeholder: test_images,
            output_placeholder: test_labels
        }

        test_accuracy, test_cost = sess.run(
            [accuracy, cost], feed_dict=feed)
        print("Test batch {:3}, Accuracy: {:3.1%}, Cost: {}"
                .format(batch_index, test_accuracy, test_cost))

        total_accuracy += test_accuracy
        total_cost += test_cost
        batches = batch_index + 1

    overall_accuracy = total_accuracy / batches
    overall_cost = total_cost / batches

    print("Total test accuracy: {:3.1%}".format(overall_accuracy))

    return overall_accuracy, overall_cost


def new_weights(shape):
    global WEIGHT_COUNTER
    randomized_tensor = tf.random_normal(
        shape=shape, dtype=tf.float32, seed=RANDOM_SEED)
    weight = tf.Variable(randomized_tensor, name='w_{}'.format(WEIGHT_COUNTER))
    WEIGHT_COUNTER += 1
    return weight


def new_biases(length):
    global BIAS_COUNTER
    randomized_tensor = tf.random_normal(
        shape=[length], dtype=tf.float32, seed=(RANDOM_SEED + 1))
    bias = tf.Variable(randomized_tensor, name='b_{}'.format(BIAS_COUNTER))
    BIAS_COUNTER += 1
    return bias


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=2):
    global CONVOLUTION_COUNTER
    global POOLING_COUNTER
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME',
                         name='conv_{}'.format(CONVOLUTION_COUNTER))
    CONVOLUTION_COUNTER += 1

    layer = tf.add(layer, biases)

    layer = tf.nn.relu(layer)

    if pooling is not None and pooling > 1:
        layer = tf.nn.max_pool(value=layer, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME',
                               name='pool_{}'.format(POOLING_COUNTER))
    POOLING_COUNTER += 1

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.add(tf.matmul(input, weights), biases)
    return layer

if __name__ == '__main__':
    main()
