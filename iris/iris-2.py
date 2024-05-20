from sklearn.datasets import load_iris

iris = load_iris()
print ("iris type:\n", type(iris))
print ("iris.data type:\n", type(iris.data))
print ("iris.target type:\n", type(iris.target))
print ("iris.data shape:\n", iris.data.shape)
print ("iris.target shape:\n", iris.target.shape)


print ("iris data:\n", iris.data)
print ("iris feature_names:\n", iris.feature_names)
print ("iris target:\n", iris.target)
print ("iris target_names:\n", iris.target_names)

X = iris.data
y = iris.target






