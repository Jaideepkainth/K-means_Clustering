import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

def dist(new, old, c=1):
    return np.linalg.norm(new-old,axis=c)

data=pd.read_csv('data.txt',names=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']) #Read Data from data.txt file
data["Class"]=data["Class"].replace(["Iris-setosa","Iris-versicolor","Iris-virginica"],[1,2,3]) #Changing Class names to numeric value
matrix=data.values
x=matrix[:,0:4]
y=matrix[:,4]
print("Enter number of clusters: ")
k=int(input())
centroids=[]
for i in range(k):
	centroids.append(np.random.randint(0, np.max(x), size = 4))
centroids=np.array(centroids, dtype=np.float)
print("Randomly Chosen centroids are:",centroids)
centroids_shape=centroids.shape
old_centroids=np.zeros(centroids_shape)
deviation_centroids=dist(centroids, old_centroids, None)
clusters=np.zeros(len(x))
while deviation_centroids!=0:
	for i in range(len(x)):
		distance_centroids=dist(x[i], centroids)
		distance_min=np.argmin(distance_centroids)
		clusters[i]=distance_min
	for i in range(k):
		print("Number of Points in Cluster ",i)
		print(np.count_nonzero(clusters == i))
	old_centroids=deepcopy(centroids)
	for i in range(k):
		pts=[]
		for m in range(len(x)):
			if i==clusters[m]:
				pts.append(x[m])
		if(len(pts)!= 0):
			centroids[i]=np.mean(pts, axis=0)
		else:
			cent=np.random.randint(0, np.max(x), size = 4)
			centroids[i]=np.array(cent, dtype=np.float)
	print("New centroids are: ",centroids)
	deviation_centroids=dist(centroids, old_centroids, None)
figure=pyplot.figure()
ax=figure.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,2], x[:,3],c=clusters,edgecolor='k',marker='o')
ax.set_xlabel('Sepal_Length')
ax.set_ylabel('Petal_Length')
ax.set_zlabel('Petal_Width')
pyplot.show()