import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

data = pd.read_csv("extract_feature/data_train.csv", header=None)
label = pd.read_csv("extract_feature/data_train_id_class.csv", header=None)

data_new = pd.concat([label,data], axis = 1)
print(data)
print(label)
print(data_new)

# plt.scatter(data['Satisfaction'], data['Loyalty'])
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()

#Selecting Features
# x = data.copy()

#Clustering
kmeans=KMeans(121)
kmeans.fit(data)

#Cluster result
clusters = data_new.copy()
clusters['cluster_pred']=kmeans.fit_predict(data)
print(clusters)
#plot
# plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()

#standardizing the variables
# x_scaled= preprocessing.scale(x)
# print(x_scaled)

#Take advantage of the elbow method
# wcss=[]

# for i in range(1,30):
# 	kmeans=KMeans(i)
# 	kmeans.fit(x_scaled)
# 	wcss.append(kmeans.inertia_)

# print(wcss)

#visualizing the elbow
# plt.plot(range(1,30),wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# kmeans_new = KMeans(4)
# kmeans.fit(x_scaled)
# cluster_new=x.copy()
# cluster_new['cluster_pred']=kmeans_new.fit_predict(x_scaled)
# print(cluster_new)

# plt.scatter(cluster_new['Satisfaction'],cluster_new['Loyalty'], c=cluster_new['cluster_pred'], cmap='rainbow')
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()