from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from connection import myDatabase

obj_DB = myDatabase('localhost', 'root', '', 'datatrain')

id_train, data_train = obj_DB.ALL_DATA_FEATURE('data_feature')
id_centroid, data_centroid = obj_DB.ALL_DATA_FEATURE('data_centroid')

# print(data_train.shape)
# print(id_train.shape)
# print(data_centroid.shape)
# print(id_centroid.shape)

data = data_train
category = id_centroid
# Number of clusters
k = id_centroid.shape[0]
# Number of training data
n = data_train.shape[0]
# Number of features in the data
c = data_train.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
centers = data_centroid

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

data.shape
clusters = np.zeros(n)
distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)

print(centers_new.shape)
print(clusters.shape)
print(clusters)

distances2 = np.zeros((n,k))
clusters2 = np.zeros(n)
for i in range(k):
	distances2[:,i] = np.linalg.norm(data - centers_new[i], axis=1)
	# Assign all training data to closest center
clusters2 = np.argmin(distances2, axis = 1)

print(clusters2.shape)
print(clusters2)

## convert your array into a dataframe
df = pd.DataFrame(clusters2)
## save to xlsx file
filepath = 'label_kMeans2.xlsx'
df.to_excel(filepath, index=False)