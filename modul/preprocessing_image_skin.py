import argparse
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from preprocessing import preprocessing_face_skin

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputImage', required=True, help='path image file')
	ap.add_argument('-o', '--outImage', required=True, help='path of output image file')
	args = ap.parse_args()

	preprocessing_face_skin(args.inputImage, args.outImage)

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python preprocessing_image_skin.py -i "D:/PYTHON_PCD/dataset_01" -o "D:/PYTHON_PCD/dataset_out_skin"
	# 
	#############################################################################################################
	
if __name__ == '__main__':
	main()