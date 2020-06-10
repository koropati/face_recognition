import sys
import argparse
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from classifier import K_MEANS
from connection import myDatabase

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--dbName', required=True, help='database_name')
	# ap.add_argument('-o', '--output', required=True, help='output excel file')
	args = ap.parse_args()

	obj_DB = myDatabase(host='localhost',user='root',pwd='',dbname=args.dbName)
	myDB, myCursor = obj_DB.hubungkan()
	print("MELOAD DATASET ..")
	id_class, id_subclass, datatrain = obj_DB.ALL_DATA_FEATURE('data_feature')
	id_class_centroid, id_subclass_centroid, datatrain_centroid = obj_DB.ALL_DATA_FEATURE('data_centroid_subclass')
	nilaiKluster = datatrain_centroid.shape[0]
	print("MEMPROSES KMEANS")
	obj_KMEAN = K_MEANS(dataset=datatrain, dataset_label=id_subclass, centroid=datatrain_centroid,k=nilaiKluster)
	data_terdekat,label_terdekat, last_center, label_kmeans=obj_KMEAN.clustering(n_closest=4, output_file="D:/PYTHON_PCD/data_clustering.csv", output_file2="D:/PYTHON_PCD/label_clustering.csv")
	print(len(data_terdekat))
	# print(len(data_terdekat[0]))
	print(len(label_terdekat))
	print(len(last_center))
	print(len(label_kmeans))
	print("SELESAI")


	

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python proses_kmeans.py -d "datatrain"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()