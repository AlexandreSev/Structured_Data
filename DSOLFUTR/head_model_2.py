#coding:utf-8

import tensorflow as tf
import numpy as np

from .convolution_part import convolutional_part
from .ngrams import get_ngrams, get_dict_ngrams



#        self.dict_ngrams = get_dict_ngrams(self.ngrams)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)



class second_model():
    """
    """

    def __init__(self, input_shape=(None, 32, 32, 1), learning_rate=1e-4):
        
        self.ngrams = get_ngrams()

        self.output_size = len(self.ngrams)

        self.cnn = convolutional_part(input_shape)

        self.W_h = weight_variable(shape=(4096, 256))
        self.b_h = bias_variable(shape=[256])
        
        self.input = tf.reshape(self.cnn.maxpool3, [-1, 4096])

        self.h = tf.nn.relu(tf.matmul(self.input, self.W_h) + self.b_h)

        self.W_o = weight_variable(shape=(256, self.output_size))
        self.b_o = bias_variable(shape=[self.output_size])

        self.output = tf.sigmoid(tf.matmul(self.h, self.W_o) + self.b_o) 


        self.target = tf.placeholder(tf.float32, shape=(None, self.output_size)) 

        self.create_train(learning_rate=learning_rate)
    

    def predict(self, x, sess, seuil=0.5):
        feed_dict = {self.cnn.x: x}
        predicted = sess.run(self.output, feed_dict=feed_dict)
        return (predicted > seuil).astype(np.int)

    def create_train(self, learning_rate=0.001):
        self.loss = tf.nn.l2_loss(self.output - self.target)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        

    def f_train_step(self, x, target, sess):
        feed_dict = {self.cnn.x: x, self.target: target}
        loss_score = (sess.run(self.loss, feed_dict=feed_dict))
        sess.run(self.train_step, feed_dict=feed_dict)
        print("Loss: %s"%loss_score)

if __name__== '__main__':
    from .settings import training_directory
    from os.path import join as pjoin
    a = second_model()
    test = np.load(pjoin(training_directory, 'test.npy')).reshape(1,32,32,3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a.predict(test)
