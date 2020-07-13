import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

#print(type(mnist))

#print(mnist.train.images)
#print(mnist.train.num_examples)
#print(mnist.train.images.shape)

single_img = mnist.train.images[1].reshape(28,28)
plt.imshow(single_img,cmap="gist_gray")
plt.show()

#print(single_img.min())
#print(single_img.max())

x = tf.placeholder(tf.float32,shape=[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b
y_true = tf.placeholder(tf.float32,shape=[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        
        batch_x,batch_y = mnist.train.next_batch(100)

        sess.run(train,feed_dict = {x:batch_x,y_true:batch_y})
    
    correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))

    acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))