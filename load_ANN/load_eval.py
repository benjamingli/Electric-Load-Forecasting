# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import load_inference
import load_train
import numpy as np
import pretreat

EVAL_INTERVAL_SECS = 5
DATA_PATH = '/home/songling/load_ANN/pretrate_single.csv'

def evaluate(filename):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [None, load_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, load_inference.OUTPUT_NODE], name='y-input')
        xs,ys = pretreat.dofile(filename, 0, 800)
        validate_feed = {x: xs,y_: ys}

        y = load_inference.inference(x, None)
        #MAPE
        sub = tf.reduce_mean(tf.abs(tf.subtract(y_,y)))
        mape_sum = tf.div(tf.abs(tf.subtract(y_,y)),y_)
        mape = tf.reduce_mean(mape_sum)

        variable_averages = tf.train.ExponentialMovingAverage(
                load_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                        load_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path\
                        .split('/')[-1].split('-')[-1]
                MAPE, subb = sess.run([mape,sub], feed_dict=validate_feed)
                print subb
                print("After %s training steps, MAPE = %g" % (global_step,MAPE))
            else:
                print('No checkpoint file found')
                return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    evaluate(DATA_PATH)

if __name__ == '__main__':
    tf.app.run()
