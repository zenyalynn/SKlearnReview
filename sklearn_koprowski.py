# Zenya Koprowski
# I pledge my honor that I have abided by the Stevens Honor System
# FE 595 Fall 2020

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ~~~ Part One: ~~~
# Used help from this website
# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

# Upload the Boston data set from SKLearn
boston_data = load_boston()

# Put data into a pandas data frame where predic are the predictors and target is the price of the homes
predic = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
target = pd.DataFrame(boston_data.target, columns=["MEDV"])

# Set x and y variables for the linear regression, we're choosing 'MEDV; because those are the values of the house
x = predic
y = target["MEDV"]
lr = LinearRegression()
lr.fit(x, y)

# created a pandas table to put all the coefficients in using help from https://stackoverflow.com/
# questions/34649969/how-to-find-the-features-names-of-the-coefficients-using-scikit-linear-regressio

table = pd.DataFrame(list(boston_data.feature_names)).copy()
table.insert(len(table.columns), "Coefs", lr.coef_.transpose())
table = table.sort_values(by=["Coefs"], ascending=False)
print("Impact of each coefficient on the value of a house:\n", table)

# ~~~ Part Two: ~~~
# Used help from these two webites for this question - https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
# https://predictivehacks.com/k-means-elbow-method-code-for-python/
iris_data = load_iris()
XI = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Here we are running 1-15 different clusters and seeing which one has the lowest SSE
error = []
for n in range(1, 15):
    kmeans = KMeans(n_clusters=n).fit(XI)
    error.append(kmeans.inertia_)

# Here we plot to solve for the optimal amount of cluster's
# We see that there is an elbow between 2 and 3 clusters, I would say 3 clusters is the best because there is a drastic
# drop off between the 2nd ad 3rd plot point
plt.plot(range(1, 15), error, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('SSE (Disortion)')
plt.title('The Elbow Method w/ Iris Data Set')
plt.show()

# Now we do a similar process with the wine data set
wine_data = load_wine()
XW = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

error2 = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, max_iter=1000).fit(XW)
    error2.append(kmeans.inertia_)

# Plot the SSE's with the cluser amount to give us the elbow graph
# We see that there is an elbow between 2/3/4 clusters, however, the graph's elbow is most distinct at the 4th cluster
# So, there should be 4 clusters with the wine data set
plt.plot(range(1, 10), error2, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('SSE (Disortion)')
plt.title('The Elbow Method w/ Wine Data Set')
plt.show()
