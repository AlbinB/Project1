"""
Project 1

At the end you should see something like this
Step Count:1000
Training accuracy: 0.8999999761581421 loss: 0.42281264066696167
Test accuracy: 0.8199999928474426 loss: 0.4739704430103302

play around with your model to try and get an even better score
"""

import tensorflow as tf
import dataUtils


training_data, training_labels = dataUtils.readData("project1trainingdata.csv")
test_data, test_labels = dataUtils.readData("project1testdata.csv")

# label placeholder
label_placeholder = tf.placeholder(tf.float32, shape=[None, 2])


# Tensorflow placeholder
input_placeholder = tf.placeholder(tf.float32, shape=[None, 113])

## Neural network hidden layers

"""
#Long Way
weight1 = tf.get_variable("weight1", shape=[113, 226], initializer=tf.contrib.layers.xavier_initializer())
bias1 = tf.get_variable("bias1", shape=[226], initializer=tf.contrib.layers.xavier_initializer())
hidden_layer1 = tf.nn.relu(tf.matmul(input_placeholder, weight1)+bias1)

weight2 = tf.get_variable("weight2", shape=[226, 150], initializer=tf.contrib.layers.xavier_initializer())
bias2 = tf.get_variable("bias2", shape=[150], initializer=tf.contrib.layers.xavier_initializer())
hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, weight2)+bias2)

weight3 = tf.get_variable("weight3", shape=[150, 100], initializer=tf.contrib.layers.xavier_initializer())
bias3 = tf.get_variable("bias3", shape=[100], initializer=tf.contrib.layers.xavier_initializer())
hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, weight3)+bias3)
###
"""
###Short Way
hidden_layer1 = tf.nn.dropout(tf.layers.dense(tf.layers.batch_normalization(input_placeholder, training=True),
                                              113, activation=tf.nn.relu), keep_prob=0.9)
hidden_layer2 = tf.nn.dropout(tf.layers.dense(tf.layers.batch_normalization(hidden_layer1, training=True),
                                              226, activation=tf.nn.relu), keep_prob=0.9)
hidden_layer3 = tf.nn.dropout(tf.layers.dense(tf.layers.batch_normalization(hidden_layer2, training=True),
                                              226, activation=tf.nn.relu), keep_prob=0.9)
hidden_layer4 = tf.nn.dropout(tf.layers.dense(tf.layers.batch_normalization(hidden_layer3, training=True),
                                              113, activation=tf.nn.relu), keep_prob=0.9)
hidden_layer5 = tf.nn.dropout(tf.layers.dense(tf.layers.batch_normalization(hidden_layer4, training=True),
                                              60, activation=tf.nn.relu), keep_prob=0.9)

# logits/Output layer
logits = tf.nn.softmax(tf.layers.dense(hidden_layer5, 2, activation=None))

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_placeholder, logits=logits))

# back-propagation algorithm
train = tf.train.AdamOptimizer().minimize(loss)
accuracy = dataUtils.accuracy(logits, label_placeholder)

# summaries
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Make tensorflow session
with tf.Session() as sess:
    ## Initialize variables
    sess.run(tf.global_variables_initializer())
    # Add writer
    summary_writer = tf.summary.FileWriter("/tmp/graph/1", sess.graph)
    step_count = 0
    while True:
        step_count += 1

        batch_training_data, batch_training_labels = \
            dataUtils.getBatch(data=training_data,
                               labels=training_labels,
                               batch_size=500)

        # train network
        training_accuracy, training_loss, logits_output, _ = \
            sess.run([accuracy, loss, logits, train],
                     feed_dict={input_placeholder: batch_training_data,
                                label_placeholder: batch_training_labels})


        # every 100 steps check accuracy
        if step_count % 100 == 0:
            batch_test_data, batch_test_labels = \
                dataUtils.getBatch(data=test_data,
                                   labels=test_labels,
                                   batch_size=500)

            test_accuracy, test_loss, logits_output, _, summary_merged, = \
                sess.run([accuracy, loss, logits, train, merged],
                     feed_dict={input_placeholder: batch_test_data,
                                label_placeholder: batch_test_labels})

            # save model
            save_path = saver.save(sess, "/Users/albin/DevProject/Project1/models/model{}.ckpt".format(step_count))

            # save summary
            summary_writer.add_summary(summary_merged, step_count)

            print("Logist {}". format(logits_output))
            print("Step Count:{}".format(step_count))
            print("Training accuracy: {} Training loss: {}".format(training_accuracy, training_loss))
            print("Test accuracy: {} Test loss: {}".format(test_accuracy, test_loss))

        # stop training after 1,000 steps
        if step_count > 1000:
            print("Model saved in path: %s" % save_path)
            print("Tensorboard logs in /tmp/graph/1")
            break
