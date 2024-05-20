#Iris Dataset Testing

from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

col_names = iris.feature_names
target_names = iris.target_names

print ("Column  Names", col_names)
print ("Target Names", target_names)


#Plot
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

plt.show()

