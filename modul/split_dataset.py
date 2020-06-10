import argparse
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import split_img

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-src', '--srcDir', required=True, help='path image file')
	ap.add_argument('-train', '--dirTrain', required=True, help='path of output image file')
	ap.add_argument('-test', '--dirTest', type=float, required=True, help='scale factor (default 1.3)')
	args = ap.parse_args()

	split_img(args.srcDir, args.dirTrain, args.dirTest)

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python split_dataset.py -src "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_hasil_preprocessing" -train "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_train" -test "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/dataset_test"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()