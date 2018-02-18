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


# Build tensorflow blueprint
## Tensorflow placeholder
input_placeholder = None # replace with your code
## Neural network hidden layers


## Logit layer
logits = None # replace with your code


## label placeholder
label_placeholder = None # replace with your code

## loss function
loss = None # replace with your code
## backpropagation algorithm
train = None # replace with your code

accuracy = dataUtils.accuracy(logits, label_placeholder)

## Make tensorflow session
with tf.Session() as sess:
    ## Initialize variables
    #sess.run(# your code)


    step_count = 0
    while True:
        step_count += 1

        batch_training_data, batch_training_labels = dataUtils.getBatch(data=training_data, labels=training_labels, batch_size=100)

        # train network
        #training_accuracy, training_loss, _ = sess.run(# your code)

        # every 10 steps check accuracy
        if step_count % 10 == 0:
            batch_test_data, batch_test_labels = dataUtils.getBatch(data=test_data, labels=test_labels,
                                                                            batch_size=100)
            #test_accuracy, test_loss = sess.run(# your code)

            print("Step Count:{}".format(step_count))
            print("Training accuracy: {} loss: {}".format(training_accuracy, training_loss))
            print("Test accuracy: {} loss: {}".format(test_accuracy, test_loss))


        # stop training after 1,000 steps
        if step_count > 1000:
            break
