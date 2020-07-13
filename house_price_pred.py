import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

housing = pd.read_csv('house_pricing.csv')
#print(housing.describe().transpose)

x_data = housing.drop(['medianHouseValue'],axis=1)
y_val = housing['medianHouseValue']

x_train, x_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=0)
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = pd.DataFrame(data=scaler.transform(x_train),columns=x_train.columns,index=x_train.index)
#print(x_train.head())
x_test = pd.DataFrame(data=scaler.transform(x_test),columns=x_test.columns,index=x_test.index)
#print(x_test.head())
#print(housing.columns)

age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age,rooms,bedrooms,pop,households,income]
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)

model.train(input_fn=input_func,steps=25000)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False)

pred_out = model.predict(pred_input_func)

predictions = list(pred_out)
#print(predictions)
final_pred = []
for pred in predictions:
    final_pred.append(pred['predictions'])

print(mean_squared_error(y_test,final_pred)**0.5)





