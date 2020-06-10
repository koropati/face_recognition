import sys
import argparse
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from classifier import K_MEANS_TO_DB

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--dbName', required=True, help='database_name')
	ap.add_argument('-n', '--nClosest', type=int, required=True, help='number of closest data to centroid')
	ap.add_argument('-data', '--dataPath', required=True, help='path for store data.csv')
	ap.add_argument('-label', '--labelPath', required=True, help='path for store label.csv')

	args = ap.parse_args()

	obj_kmeans = K_MEANS_TO_DB(database = args.dbName, n_closest = args.nClosest,output_data = args.dataPath, output_label = args.labelPath)
	obj_kmeans.clustering()

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python proses_kmeans_DB.py -d "dataset_face" -n 4 -data "D:/PYTHON_PCD/file_csv/data_cluster.csv" -label "D:/PYTHON_PCD/file_csv/data_cluster_label.csv"
	# python proses_kmeans_DB.py -d "datatrain" -n 4 -data "D:/PYTHON_PCD/file_csv/data_cluster2.csv" -label "D:/PYTHON_PCD/file_csv/data_cluster_label2.csv"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()