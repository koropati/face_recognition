import sys, cv2, argparse
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import combineDetector
from face_cnn import recognize_face


def predict(img_in, img_out_path):
	img_asli = cv2.imread(img_in)

	obj_detection = combineDetector(url=img_in, scaleFactor=1.3, minNeighbors=6)
	x1, y1, x2, y2, img_out = obj_detection.get_face()

	obj_recognize = recognize_face(img_input=img_out, model_path='D:/PYTHON_PCD/data_cnn/model_cnn.h5', label_trains='D:/PYTHON_PCD/data_cnn/name_class.csv')
	label, name_class = obj_recognize.predict()

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.rectangle(img_asli, (x1, y1), (x2, y2), (0,255,0), 4)
	cv2.putText(img_asli, name_class, (x1, (y2+200)), font, 4, (0,255,0), 18)
	cv2.imwrite(img_out_path, img_asli)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputImage', required=True, help='Input File Image')
	ap.add_argument('-o', '--outputImage', required=True, help='Output File Image')
	args = ap.parse_args()

	predict(args.inputImage, args.outputImage)
	print('selesai!')

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python predict_cnn.py -i "D:/PYTHON_PCD/1915051088-depan_7.JPG" -o "D:/PYTHON_PCD/hasil_predict_cnn.jpg"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()