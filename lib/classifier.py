import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from datetime import datetime
import os.path
import xlsxwriter
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import printProgressBar, append_list_as_row
from connection import myDatabase

class KNN_CLASSIFIER(object):
	def __init__(self, dataset, dataset_label, k, test_size):
		self.dataset = dataset
		self.dataset_label = dataset_label
		self.test_size = test_size
		self.k = k
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.dataset_label, test_size=self.test_size)
		self.knn = KNeighborsClassifier(n_neighbors=self.k, weights='uniform', algorithm='auto', metric='euclidean')

	def run(self):
		#datatrain, label_datatrain
		self.knn.fit(self.X_train, self.y_train)

	def testing(self, output_file, kFold):
		if os.path.isfile(output_file):
			print("File: "+output_file+" is Exist")
		else:
			print("File: "+output_file+" is not Exist")
			print("Creating New File")
			workbook = xlsxwriter.Workbook(output_file)
			workbook.close()
		#untuk mengetahui seberapa akurat clasifiernya
		print("Process, prediction...")
		y_pred = self.knn.predict(self.X_test)
		confus_matrix = confusion_matrix(self.y_test, y_pred)
		report = classification_report(self.y_test, y_pred, output_dict=True)
		df = pd.DataFrame(report).transpose()

		print("Process, K-Fold Validation, using K="+str(kFold))
		cv_scores = cross_val_score(self.knn, self.X_train, self.y_train, cv=kFold)
		df2 = pd.DataFrame(cv_scores).transpose()

		# Menyimpan hasil ke excel file dan append Sheet nya
		now = datetime.now()
		tanggal_uji = now.strftime("%d-%m-%Y_%H-%M-%S")
		with pd.ExcelWriter(output_file, mode='a', engine="openpyxl") as writer:
			df.to_excel(writer, sheet_name=tanggal_uji+"_K_"+str(self.k))
			df2.to_excel(writer, sheet_name=tanggal_uji+"_K_"+str(self.k)+"_Kfold_"+str(kFold))
		print(classification_report(self.y_test, y_pred))
		print('K-fold-validation scores mean:{}'.format(np.mean(cv_scores)))

	def prediction(self, newArray):
		#input data fitur baru (newArr) untuk mengetahui data baru tersebut masuk ke class mana.
		label_prediction = self.knn.predict(newArray)
		return label_prediction

class KNN_CLASSIFIER_DB(object):
	def __init__(self, database, k, test_size):

		self.obj_DB = myDatabase(host='localhost',user='root',pwd='',dbname=database)
		self.myDB, self.myCursor = self.obj_DB.hubungkan()
		print("MELOAD DATASET ..")

		self.dataset, self.dataset_label = self.obj_DB.READ_DATA_KMEANS()
		self.dataset_label = self.dataset_label.ravel()
		print(self.dataset.shape)
		print(self.dataset_label.shape)
		self.test_size = test_size
		self.k = k
		
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.dataset_label, test_size=self.test_size)
		self.knn = KNeighborsClassifier(n_neighbors=self.k, weights='uniform', algorithm='auto', metric='euclidean')

	def run(self):
		#datatrain, label_datatrain
		self.knn.fit(self.X_train, self.y_train)

	def testing(self, output_file, kFold):
		print("MEMPROSES KNN..")
		if os.path.isfile(output_file):
			print("File: "+output_file+" is Exist")
		else:
			print("File: "+output_file+" is not Exist")
			print("Creating New File")
			workbook = xlsxwriter.Workbook(output_file)
			workbook.close()
		#untuk mengetahui seberapa akurat clasifiernya
		print("Process, prediction...")
		y_pred = self.knn.predict(self.X_test)
		confus_matrix = confusion_matrix(self.y_test, y_pred)
		report = classification_report(self.y_test, y_pred, output_dict=True)
		df = pd.DataFrame(report).transpose()

		print("Process, K-Fold Validation, using K="+str(kFold))
		cv_scores = cross_val_score(self.knn, self.X_train, self.y_train, cv=kFold)
		df2 = pd.DataFrame(cv_scores).transpose()

		# Menyimpan hasil ke excel file dan append Sheet nya
		now = datetime.now()
		tanggal_uji = now.strftime("%d-%m-%Y_%H-%M-%S")
		with pd.ExcelWriter(output_file, mode='a', engine="openpyxl") as writer:
			df.to_excel(writer, sheet_name=tanggal_uji+"_K_"+str(self.k))
			df2.to_excel(writer, sheet_name=tanggal_uji+"_K_"+str(self.k)+"_Kfold_"+str(kFold))
		print(classification_report(self.y_test, y_pred))
		print('K-fold-validation scores mean:{}'.format(np.mean(cv_scores)))

	def testing_withOtherTest(self, database_test, output_file, kFold):
		obj_DB_test = myDatabase(host='localhost',user='root',pwd='',dbname=database_test)
		myDB_test, myCursor_test = obj_DB_test.hubungkan()
		print("MELOAD DATASET TEST..")
		id_class_test, id_subclass_test, data_test = obj_DB_test.ALL_DATA_FEATURE(tabel="data_feature")

		new_dataset = np.concatenate((self.dataset,data_test))
		new_dataset_label = np.concatenate((self.dataset_label, id_class_test.ravel()))
		print("MEMPROSES KNN..")
		X_train_new, X_test_new, y_train_new, y_test_new=train_test_split(new_dataset,new_dataset_label, test_size=self.test_size)
		new_knn = KNeighborsClassifier(n_neighbors=self.k, weights='uniform', algorithm='auto', metric='euclidean')
		new_knn.fit(X_train_new, y_train_new)
		if os.path.isfile(output_file):
			print("File: "+output_file+" is Exist")
		else:
			print("File: "+output_file+" is not Exist")
			print("Creating New File")
			workbook = xlsxwriter.Workbook(output_file)
			workbook.close()
		#untuk mengetahui seberapa akurat clasifiernya
		print("Process, prediction...")
		y_pred_new = new_knn.predict(X_test_new)
		confus_matrix = confusion_matrix(y_test_new, y_pred_new)
		report = classification_report(y_test_new, y_pred_new, output_dict=True)
		df = pd.DataFrame(report).transpose()

		print("Process, K-Fold Validation, using K="+str(kFold))
		cv_scores = cross_val_score(new_knn, X_train_new, y_train_new, cv=kFold)
		df2 = pd.DataFrame(cv_scores).transpose()

		# Menyimpan hasil ke excel file dan append Sheet nya
		now = datetime.now()
		tanggal_uji = now.strftime("%d-%m-%Y_%H-%M-%S")
		with pd.ExcelWriter(output_file, mode='a', engine="openpyxl") as writer:
			df.to_excel(writer, sheet_name=tanggal_uji+"_K_"+str(self.k))
			df2.to_excel(writer, sheet_name=tanggal_uji+"_K_"+str(self.k)+"_Kfold_"+str(kFold))
		print(classification_report(y_test_new, y_pred_new))
		print('K-fold-validation scores mean:{}'.format(np.mean(cv_scores)))

	def prediction(self, newArray):
		#input data fitur baru (newArr) untuk mengetahui data baru tersebut masuk ke class mana.
		label_prediction = self.knn.predict(newArray)
		return label_prediction

class K_MEANS(object):
	def __init__(self, dataset, dataset_label, centroid, k):
		self.dataset = np.array(dataset)
		self.dataset_label = np.array(dataset_label)
		self.centroid = np.array(centroid)
		self.k = k
		self.km = KMeans(algorithm='auto',n_init=1, init = self.centroid,n_clusters=self.k).fit(self.dataset)

	def clustering(self, n_closest, output_file, output_file2):
		last_center = self.km.cluster_centers_
		label_kmeans = self.km.labels_
		last_label = []
		last_data = []
		count=0
		label_class = 0

	

		# last_data.extend(last_center[n])
		# last_label.append(n)
		
		data=[]
		label=[]

		count=0
		label_index = 0
		printProgressBar(0, self.k, prefix = 'Progress:', suffix = 'Complete', length = 50)
		for i in range(self.k):
			d = self.km.transform(self.dataset)[:, i]
			ind = np.argsort(d)[::][:n_closest]
			data_terdekat = list(self.dataset[ind])
			# label_terdekat = list(label_kmeans[ind])
			if count >= 3:
				label_index += 1
				count = 0
			
			append_list_as_row(output_file,last_center[i])
			append_list_as_row(output_file2,[label_index])
			for x in range(len(data_terdekat)):
				append_list_as_row(output_file,data_terdekat[x])
				append_list_as_row(output_file2,[label_index])

			printProgressBar(i + 1, self.k, prefix = 'Progress:', suffix = 'Complete', length = 50)
			count += 1

		return data, label, last_center, label_kmeans
		# return last_data, last_label


class K_MEANS_TO_DB(object):
	def __init__(self, database, n_closest, output_data, output_label):
		self.obj_DB = myDatabase(host='localhost',user='root',pwd='',dbname=database)
		self.myDB, self.myCursor = self.obj_DB.hubungkan()
		print("MELOAD DATASET ..")
		id_class, id_subclass, datatrain = self.obj_DB.ALL_DATA_FEATURE('data_feature')
		id_class_centroid, id_subclass_centroid, datatrain_centroid = self.obj_DB.ALL_DATA_FEATURE('data_centroid_subclass')
		
		print("MEMPROSES KMEANS..")
		self.n_closest = n_closest
		self.output_data = output_data
		self.output_label = output_label
		self.dataset = np.array(datatrain)
		self.dataset_label = np.array(id_subclass)
		self.centroid = np.array(datatrain_centroid)
		self.k = datatrain_centroid.shape[0]
		self.km = KMeans(algorithm='auto',n_init=1, init = self.centroid,n_clusters=self.k).fit(self.dataset)

	def clustering(self):
		last_center = self.km.cluster_centers_
		label_kmeans = self.km.labels_
		last_label = []
		last_data = []
		count=0
		label_class = 0


		data=[]
		label=[]

		count=0
		label_index = 0
		printProgressBar(0, self.k, prefix = 'Progress:', suffix = 'Complete', length = 50)
		for i in range(self.k):
			d = self.km.transform(self.dataset)[:, i]
			ind = np.argsort(d)[::][:self.n_closest]
			data_terdekat = list(self.dataset[ind])
			# label_terdekat = list(label_kmeans[ind])
			if count >= 3:
				label_index += 1
				count = 0
			
			append_list_as_row(self.output_data,last_center[i])
			append_list_as_row(self.output_label,[label_index])
			self.obj_DB.UPLOAD_DATA_KMEANS(label_index, last_center[i], 0)
			for x in range(len(data_terdekat)):
				append_list_as_row(self.output_data,data_terdekat[x])
				append_list_as_row(self.output_label,[label_index])
				self.obj_DB.UPLOAD_DATA_KMEANS(label_index, data_terdekat[x], 0)
			printProgressBar(i + 1, self.k, prefix = 'Progress:', suffix = 'Complete', length = 50)
			count += 1

		print("Selesai!")
		# return data, label, last_center, label_kmeans
		# return last_data, last_label
