import argparse
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from preprocessing import preprocessing_face

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputImage', required=True, help='path image file')
	ap.add_argument('-o', '--outImage', required=True, help='path of output image file')
	ap.add_argument('-s', '--scaleFactor', type=float, required=True, help='scale factor (default 1.3)')
	ap.add_argument('-n', '--neighboor', type=int, required=True, help='neighboor (default 6)')
	args = ap.parse_args()

	preprocessing_face(args.inputImage, args.outImage, args.scaleFactor, args.neighboor)

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python preprocessing_image.py -i "D:/PYTHON_PCD/dataset_01" -o "D:/PYTHON_PCD/dataset_out_combineDetector" -s 1.3 -n 6
	# 
	#############################################################################################################
	
if __name__ == '__main__':
	main()