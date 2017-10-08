# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 4  #9个特征
OUTPUT_NODE = 1 #输出是负荷预测值，所以是1
LAYER1_NODE = 500 #实测5个特征时较好取值50左右

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
            "weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
                [INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(
                "biases", [LAYER1_NODE],
                initializer = tf.constant_initializer(0.0))
        #layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        layer1 = tf.tanh(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
                [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
                "biases", [OUTPUT_NODE],
                initializer = tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
'''
x = tf.placeholder(
        tf.float32, [None, INPUT_NODE], name='x-input')
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
y = inference(x, regularizer)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    xs = [[4,2016,3,31,9.75,0,22.4,13.8,3]]
    yreal = sess.run(y,feed_dict={x: xs})
    print yreal
'''

