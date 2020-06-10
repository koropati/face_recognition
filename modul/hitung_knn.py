import sys
import argparse
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from classifier import KNN_CLASSIFIER
from processing import read_csv_float, read_csv_int, read_csv_string

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--dataset', required=True, help='Source Dataset')
	ap.add_argument('-l', '--datasetLabel', required=True, help='Source Dataset Label')
	ap.add_argument('-k', '--neighboor', type=int, required=True, help='neighboor (default 5)')
	# ap.add_argument('-o', '--output', required=True, help='output excel file')
	args = ap.parse_args()

	print("MELOAD DATASET ..")
	dataset = read_csv_float(args.dataset)
	print("MELOAD DATASET LABEL ..")
	dataset_label = read_csv_int(args.datasetLabel)
	print("PROCESSING KNN")
	obj_KNN = KNN_CLASSIFIER(dataset=dataset, dataset_label=dataset_label, k=args.neighboor, test_size=0.30)
	obj_KNN.run()
	obj_KNN.testing(output_file="D:/PYTHON_PCD/data_clustering.csv", kFold=6)
	print("SELESAI!")

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python hitung_knn.py -d "D:/PYTHON_PCD/dataset.csv" -l "D:/PYTHON_PCD/dataset_id_class.csv" -k 5
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()