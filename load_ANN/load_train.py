# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import load_inference
import pretreat
import pandas as pd
from numpy import *

BATCH_SIZE = 400  #实测当一个变电站时（大约3000个数据，取400最佳，loss约37）
DATA_SIZE = 2928
TRAIN_DATA = int(DATA_SIZE*0.8)
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 4001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/home/songling/load_ANN/model"
MODEL_NAME = "model.ckpt"

DATA_PATH = "/home/songling/load_ANN/pretrate_single.csv"

def train(filename):
    x = tf.placeholder(
        tf.float32, [None, load_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, load_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = load_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    varibale_averages_op = variable_averages.apply(
            tf.trainable_variables())
    mse = tf.reduce_mean(tf.square(y_ - y)) #回归问题要用均方误差MSE
    loss= mse + tf.add_n(tf.get_collection('losses'))
    #MAPE
    sub = tf.reduce_mean(tf.abs(tf.subtract(y, y_)))
    mape_sum = tf.div(tf.abs(tf.subtract(y, y_)),y_)
    mape = tf.reduce_mean(mape_sum)

    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            DATA_SIZE / BATCH_SIZE,
            LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, varibale_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        x_raw, y_raw = pretreat.dofile(DATA_PATH, DATA_SIZE)
        xs = x_raw[0:TRAIN_DATA]
        ys = y_raw[0:TRAIN_DATA]
        x_test = x_raw[TRAIN_DATA:DATA_SIZE-1]
        y_test = y_raw[TRAIN_DATA:DATA_SIZE-1]

        for i in range(TRAINING_STEPS):
            start = (i * BATCH_SIZE) % TRAIN_DATA
            end = min(start+BATCH_SIZE, TRAIN_DATA)
            xx = xs[start:end]
            yy = ys[start:end]
            _,loss_value, step = sess.run([train_op, \
                    loss, global_step ],feed_dict={x: xx, y_: yy})    
            if i % 500 == 0:
                loss_555, mape_value, y_true, y_pred = sess.run([loss, mape, y_, y],\
                        feed_dict={x: x_test, y_: y_test})
                print("After %d training step(s), loss on test"
                    " data is %.4f, MAPE = %.4f" % (step, loss_555, mape_value))
                #输出预测结果
                if i >= 100: 
                    y_con = concatenate((y_true, y_pred), axis=1)
                    y_out = pd.DataFrame(y_con, columns=["true_data","pre_data"])
                    y_out.to_csv('/home/songling/load_ANN/output/steps=%d-MAPE=%.4f.csv'\
                                % (i, mape_value))

                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
                    global_step = global_step)

def main(argv=None):
    train(DATA_PATH)

if __name__ == '__main__':
    tf.app.run()
