import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.python.ops import array_ops as array_ops_
import matplotlib.pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 3000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    output,_ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])

    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    labels = tf.reshape(y, [-1])
    predictions = tf.losses.mean_squared_error(predictions, labels)

    train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_steps(),
            optimizer-'Adagrad', learning_rate=0.1)
    return predictions, loss, train_op

regressor = SKCompat(learn.Estimator(model_fn=lstm_model,model_dir="Models/model_2"))

test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
                0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
                test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

predicted = [[pred] for pred in regressor.predict(test_X)]

rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print "Mean Square Error is: %f" % rmse[0]
