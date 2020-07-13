import tensorflow as tf

hello = tf.constant("hELLO")#tensor is a name for array,tensor represents a separate data type for array
#print(hello)
world = tf.constant("World") #World is a tensor constant holding a string type of data
#print(type(hello))

with tf.Session() as sess:#Everything in Tensorflow is executed via a session, nothing can be done normally
    result = sess.run(hello+world)#b represents byte literal type of data

print(result)

var1=tf.constant(10)#Tensors of the type int32, can't be normally added
var2=tf.constant(30)

with tf.Session() as sess:
    result = sess.run(var1+var2)

print(result)

mat_1 = tf.fill((5,5),10)#A 5x5 matrix filled with all values as 10
zeros = tf.zeros((5,5))#A 5x5 matrix filled with zeros
ones = tf.ones((5,5))#A 5x5 matrix filled with ones
randn = tf.random_normal((4,4),mean=0,stddev=1.0)#A 4x4 matrix hich has random normal values and a given mean and standard deviation
randu = tf.random_uniform((4,4),minval=0,maxval=1)#A 4x4 random uniform matrix between a given minimum value and maximum value 

myops = [mat_1,zeros,ones,randn,randu]

sess = tf.InteractiveSession()#Another way to run a session continuously in Tensorflow

for op in myops:
    #print(sess.run(op))
    print(op.eval())#A way to evaluate a session in tensorflow
    print("\n")

a = tf.constant([[1,2],[3,4]])

print(a.get_shape())#Function to print matrix shape in tensorflow

b=tf.constant([[10],[100]])

print(b.get_shape()) 

result =tf.matmul(a,b)#Function to multiply appropriate dimension matrix in Tensorflow

print(sess.run(result))


