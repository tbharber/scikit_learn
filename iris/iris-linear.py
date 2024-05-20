#Iris Dataset w/ Linear regression

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

print ("Head:\n", df.head())
print ("Shape:\n", df.shape)

print ("Correlation:\n", df.corr())
print ("Describe:\n", df.describe())

df.plot.scatter(x='petal width (cm)', y='petal length (cm)', title = 'scatterplot')

#Reshape data
y = df['petal width (cm)'].values.reshape(-1,1)
x = df['petal length (cm)'].values.reshape(-1,1)

#Split data
from sklearn. model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print ("Training Data:\n")
print (x_train)
print (x_train)

#Train model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

print(regressor.intercept_)
print (regressor.coef_)

#Prediction
score = regressor.predict([[7.5]])
print ("Predict:", score)

def calc(slope,intercept,value):
    return slope * value + intercept

score = calc(regressor.coef_, regressor.intercept_, 7.5 )
print (score)


y_pred = regressor.predict(x_test)
df_pred = pd.DataFrame({'Actual' : y_test.squeeze(), 'Predicted' : y_pred.squeeze()})
print (df_pred)

#Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print (mae)
print (mse)
print (rmse)