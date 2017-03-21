#coding:utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle

from .convolution_part import convolutional_part
from ..utils.ngrams import get_ngrams, get_dict_ngrams


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

    def train(self, x, target, sess, nb_epoch=100, save=True, warmstart=False, 
              weights_path="./model2.ckpt", save_path="./model2.ckpt", training_target=None):

        prediction = self.predict(x, sess)
        print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

        saver = tf.train.Saver()

        if warmstart:
            saver.restore(sess, weights_path)
            print("Model Loaded.")
        
        accuracy = []
        for i in range(1, nb_epoch + 1):

            self.f_train_step(x, target, sess)
            print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
            
            if i % 10 == 0:
                prediction = self.predict(x, sess)
                print("accuracy: "+str(np.mean(prediction == target)))
                accuracy.append(np.mean(prediction == target))

            if save & (i % 50 == 0):
                save_path = saver.save(sess, save_path)
                print("Model saved in file: %s" % save_path)
                with open('accuracy_2.pickle', 'wb') as file:
                    pickle.dump(accuracy, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__== '__main__':
    from .settings import training_directory
    from os.path import join as pjoin
    a = second_model()
    test = np.load(pjoin(training_directory, 'test.npy')).reshape(1,32,32,3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a.predict(test)
