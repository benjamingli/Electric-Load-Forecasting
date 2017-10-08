#-*- coding:utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pretreat

learning_rate = 0.001
training_iters = 1000
batch_size = 400


data_size = 2928
n_input = 4
#用前10个数据预测下一个,第batch_size个数据，n_step个为一组，一个n_input个特征
n_steps = 40
n_hidden = 128
n_class = 1 
n_layers = 3

data_path= "/home/songling/load_RNN/pretrate_single.csv"

x = tf.placeholder("float", [None, n_steps, n_input])
y_ = tf.placeholder("float", [None, n_class])

weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_class]))}
biases = {
        'out': tf.Variable(tf.random_normal([n_class]))}

def DeepRnn(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)            
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1) 
    def lstm_cell():
        return rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    #用下面这个写法写深层神经网络，上一个方法版本不兼容，会报错
    stacked_lstm = rnn.MultiRNNCell(
        [lstm_cell() for _ in range(n_layers)])
    #定义输出列表，将不同时刻LSTM结构的输出收集起来
    outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
    #将output转化为[batch, hidden*steps]
    output = tf.reshape(tf.stack(outputs,1), [-1, n_hidden])
    return tf.nn.bias_add(tf.matmul(output,weights['out']),biases['out'])

pred = DeepRnn(x, weights, biases)
loss = tf.losses.mean_squared_error(pred, y_)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_iters):
        start = (i * batch_size) % data_size
        end = min(start+batch_size, data_size)
        if end == data_size:
            end = end - ((end - start) % n_steps)
            if end-start == 0:
                continue
        xs, ys = pretreat.dofile(data_path, start, end)
        xs = np.array(xs).reshape((-1, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: xs, y_: ys})
        if i % 10 == 0:
            loss_value = sess.run(loss, feed_dict={x: xs, y_: ys})
            print("After %d training step(s), loss on training "
                "data is %.4f" % (i, loss_value))



