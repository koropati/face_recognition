import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from connection import myDatabase

obj_DB = myDatabase('localhost', 'root', '', 'datatrain')
id_train, id_train_sub, data_train = obj_DB.ALL_DATA_FEATURE('data_feature')
id_centroid_sub, data_centroid = obj_DB.ALL_DATA_FEATURE('data_centroid_subclass')

k = id_centroid_sub.shape[0]
X = np.random.randn(10,4)
km = KMeans(algorithm='auto',init = data_centroid,n_clusters=k).fit(data_train)
last_center = km.cluster_centers_
print(km.labels_)

## convert your array into a dataframe
df = pd.DataFrame(km.labels_)
## save to xlsx file
filepath = 'label_kMeans_sub1.xlsx'
df.to_excel(filepath, index=False)


# closest, _ = pairwise_distances_argmin_min(km.cluster_centers_,X)
# print(closest)