import sys
import argparse
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from classifier import KNN_CLASSIFIER_DB

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--database', required=True, help='Database Name')
	ap.add_argument('-k', '--neighboor', type=int, required=True, help='neighboor (default 5)')
	ap.add_argument('-s', '--splitSize', type=float, required=True, help='Size of Split Image test(0.30)')
	# ap.add_argument('-o', '--output', required=True, help='output excel file')
	args = ap.parse_args()

	obj_KNN = KNN_CLASSIFIER_DB(database=args.database, k=args.neighboor, test_size=args.splitSize)
	obj_KNN.run()
	# obj_KNN.testing(output_file="D:/PYTHON_PCD/HASIL_KNN_KMEANS.xlsx", kFold=5) #pengujian kombinasi kmeans dengan semua dataset (train_test jadi satu)
	obj_KNN.testing_withOtherTest(database_test="datatest",output_file="D:/PYTHON_PCD/HASIL_KNN_KMEANS_uji_dgn_database_datatest.xlsx", kFold=5)
	print("SELESAI!")

	######################################## RUN PROGRAM BY TYPE ################################################
	##pengujian kombinasi kmeans dengan semua dataset (train_test jadi satu)
	#
	# python hitung_knn_DB.py -d "dataset_face" -k 5 -s 0.30
	# 
	#Pengujian kombinasi kmeans dengan hanya dataset train yabg di kmeans kan pengujian menggunakna dataset test (setelah di kmeans kan datatrainnya baru di gabung dengan data test) 
	# 
	# python hitung_knn_DB.py -d "datatrain" -k 5 -s 0.30
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()