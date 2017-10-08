#-*- coding:utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pretreat
import pandas as pd


learning_rate = 0.001
training_iters = 10001
batch_size = 400


data_size = 2928
train_data = int(2928*0.8)
n_input = 4
#用前10个数据预测下一个,第batch_size个数据，n_step个为一组，一个n_input个特征
n_steps = 40
n_hidden = 128
n_class = 1 
n_layers = 3

data_path= "/home/songling/load_RNN/pretrate_single.csv"

x = tf.placeholder("float", [None, n_steps, n_input])
y_ = tf.placeholder("float", [None, n_class])

x_raw,y_raw = pretreat.dofile(data_path, data_size)
xs = x_raw[0:train_data]
ys = y_raw[0:train_data]

#去零头，方便被n_steps整除
end_test = (data_size-1) - (((data_size-1) - train_data) % n_steps)
x_test = x_raw[train_data:end_test]
y_test = y_raw[train_data:end_test]

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
#MAPE
sub = tf.reduce_mean(tf.abs(tf.subtract(pred, y_)))
mape_sum = tf.div(tf.abs(tf.subtract(pred, y_)),y_)
mape = tf.reduce_mean(mape_sum)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_iters):
        start = (i * batch_size) % train_data
        end = min(start+batch_size, train_data)
        if end == train_data:
            end = end - ((end - start) % n_steps)
            if end-start == 0:
                continue
        xx = xs[start:end]
        yy = ys[start:end]
        xx = np.array(xx).reshape((-1, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: xx, y_: yy})
        if i % 500 == 0:
            x_change = np.array(x_test).reshape((-1, n_steps, n_input))
            loss_value, mape_test, y_true, y_pred =\
                    sess.run([loss, mape, y_, pred], feed_dict={x: x_change, y_: y_test})
            print("After %d training step(s), loss on test "
                "data is %.4f, MAPE = %.4f" % (i, loss_value, mape_test))
            if mape_test <= 0.060:
                #两个ndarray列合并
                y_con = np.concatenate((y_true, y_pred), axis=1)
                #输出真实值和预测值
                y_out = pd.DataFrame(y_con, columns=["true_data","pre_data"])
                y_out.to_csv('/home/songling/load_RNN/output/steps=%d-MAPE=%.4f.csv'\
                            % (i, mape_test))
                



