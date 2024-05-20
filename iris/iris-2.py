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

print (X.shape)
print (y.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

print (knn)

knn.fit(X, y)

#knn.predict([3, 5, 4, 2])

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print (knn.predict(X_new))


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print (knn.predict(X_new))

from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression()

logreg.fit(X, y)

print(logreg.predict(X_new))



