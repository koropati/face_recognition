import argparse
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import extract_feature_toCSV

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputDir', required=True, help='path for folder image file')
	ap.add_argument('-o', '--outputDir', required=True, help='path for output file')
	ap.add_argument('-sub', '--subClass', required=True, help='Subclass or not (y/n)')
	ap.add_argument('-name', '--outName', required=True, help='name of file feature')
	args = ap.parse_args()

	extract_feature_toCSV(args.inputDir, args.outputDir, args.subClass, args.outName)
	
	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python extract_fitur_csv.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_test" -o "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/hasil_extract_feature" -sub "n" -name "data_test"
	# python extract_fitur_csv.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_train" -o "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/hasil_extract_feature" -sub "n" -name "data_train"
	# 
	# 
	# python extract_fitur_csv.py -i "D:/PYTHON_PCD/dataset_clean" -o "D:/PYTHON_PCD" -sub "n" -name "dataset"
	#############################################################################################################

if __name__ == '__main__':
	main()