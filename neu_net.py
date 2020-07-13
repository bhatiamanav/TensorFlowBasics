import numpy as np 
import tensorflow as tf 

n_f = 10#Defining the numbers of features per neuron
n_d_n = 3#Defining the neuron density per layer of the network

x = tf.placeholder(tf.float32,(None,n_f))#Initialising the input variable to our neural network
b = tf.Variable(tf.zeros([n_d_n]))#Creating the bias for all our neirons

w=tf.Variable(tf.random_normal([n_f,n_d_n]))#Creating the weights for all our neurons

#y=mx+c
xw = tf.matmul(x,w)#Multiplying the input matrix with the weights

z=tf.add(xw,b)#Adding the bias of each of the perceptron

#activation function
ans = tf.sigmoid(z)

#Variable init
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    layer_out = sess.run(ans,feed_dict={x : np.random.random([1,n_f])})
    print(layer_out)
