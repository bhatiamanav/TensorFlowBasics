import numpy as np 
import tensorflow as tf 

np.random.seed(101)#Generates a random_seed with 101 values
tf.set_random_seed(101)#Sets a random seed of 101 values

rand_a = np.random.uniform(0,100,(5,5))#Genearting a Random uniform matrix
#print(rand_a)

rand_b = np.random.uniform(0,100,(5,1))

#Creating placeholders for future operations
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

#Defining functions to be used in our Perceptron's session
add=a+b
mul=a*b

with tf.Session() as sess:
    add_result = sess.run(add,feed_dict={a:rand_a,b:rand_b})#Running add function
    #Feed Dictionary is a variable of run used when placeholders are being used to pass values
    #In feed_dict, we specify the values we wish the variables in our function executed using run would take
    print(add_result)
    print("\n")

    mult=sess.run(mul,feed_dict={a:rand_a,b:rand_b})
    print(mult)