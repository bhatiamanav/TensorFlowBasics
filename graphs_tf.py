#Graphs are created on System backend when a Tensorflow process is done.
#They consist of nodes containing values and operations & edges containing results leading out of opeartion nodes
#Tensorflow is primarily used for Computation graphs
import tensorflow as tf 

var1 = tf.constant(10)
var2 = tf.constant(20)
var3 = var1+var2

with tf.Session() as sess:
    result = sess.run(var3)

print(result)
print(var3)
print(tf.get_default_graph())#Prints the address location of the default graph created on backend for all the whole tensorflow operation done in this branch

#Creates a new graph and we can print its location
g = tf.Graph()
#print(g)
with g.as_default():#Sets the graph that we created as default graph
    print(g is tf.get_default_graph())

#g1=tf.get_default_graph()#Changes g1's address to that of default graph i.e overwrites g1's address at place of default graph 
#print(g1)