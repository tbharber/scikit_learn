#Iris Dataset w/ Linear regression

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(data= iris.data, columns= iris.feature_names)

print (df.head())
print (df.shape)

print (df.plot.scatter(x='sepal length (cm)', y='petal width (cm)'))
print (df.corr())
print (df.describe())