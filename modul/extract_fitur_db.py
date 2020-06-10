import argparse
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import extract_feature_toDB
 
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputDir', required=True, help='path for folder image file')
	ap.add_argument('-id', '--idFeature', required=True, help='id Feature 1/2/3/..')
	ap.add_argument('-sub', '--subClass', required=True, help='Subclass or not (y/n)')
	ap.add_argument('-data', '--dataBase', required=True, help='name of file feature')
	args = ap.parse_args()

	extract_feature_toDB(args.inputDir, args.idFeature, args.subClass, args.dataBase)
	
	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python extract_fitur_db.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_test" -id "0" -sub "n" -data "test"
	# python extract_fitur_db.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_train" -id "0" -sub "n" -data "train"
	# 
	# 
	# python extract_fitur_db.py -i "D:/PYTHON_PCD/dataset_clean2" -id "0" -sub "n" -data "dataset"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()