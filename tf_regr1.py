import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split

x_data = np.linspace(0.0,10.0,1000000)

noise = np.random.randn(len(x_data))

#y=mx+b , b=5
y_true = (0.5*x_data)+5+noise

x_df = pd.DataFrame(data = x_data, columns=['X_data'])
y_df = pd.DataFrame(data = y_true, columns=['Y_data'])

#print(x_df.head())
#print(y_df.head())

my_data = pd.concat([x_df,y_df],axis=1)

#print(my_data.head())
#my_data.sample(n=250).plot(kind='scatter',x='X_data',y='Y_data')
#plt.show()

batch_size=8
m = tf.Variable(0.5)
b = tf.Variable(1.0)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = m*xph + b
#Cost Function
error = tf.reduce_sum(tf.square(yph-y_model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train=optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    batches = 10000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        sess.run(train,feed_dict=feed)
    
    model_m,model_b = sess.run([m,b])

print(model_m)
print(model_b)

y_hat = x_data*model_m + model_b
#my_data.sample(n=250).plot(kind='scatter',x='X_data',y='Y_data')
#plt.plot(x_data,y_hat,'g')
#plt.show()

feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns = feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true, test_size=0.3, random_state=0)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)
train_input_func=tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)

estimator.train(input_fn=input_func,steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
#print(train_metrics)
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)

input_fu_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)
pred_list = list(estimator.predict(input_fn=input_fu_predict))
#print(pred_list)

predictions = []
for x in estimator.predict(input_fn=input_fu_predict):
    predictions.append(x['predictions'])

my_data.sample(n=250).plot(kind='scatter',x='X_data',y='Y_data')
plt.plot(np.linspace(0,10,10),predictions,'r')
plt.show()
