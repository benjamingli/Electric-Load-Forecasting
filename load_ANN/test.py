import tensorflow as tf
import load_inference
import pretreat

x = tf.placeholder(
    tf.float32, [None, load_inference.INPUT_NODE], name='x-input')
with tf.Session() as sess:
    tf.global_variable_initializer().run()
    for i in range(3):
        xs , ys = pretreat.dofile('/home/songling/load_ANN/raw.csv',i,i+3)
        x = sees.run([])
