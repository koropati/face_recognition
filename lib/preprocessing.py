import sys
import os
import glob
import shutil
import cv2
import string, calendar, datetime, traceback
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import haarDetector, skinDetector, combineDetector, printProgressBar

def preprocessing_face(input_path, output_path, scaleFactor, minNeighbors):

	if not os.path.exists(output_path):
		os.mkdir(output_path)
		print("Directory "+output_path+" Created!")

	jumlah_image = sum([len(files) for r, d, files in os.walk(input_path)])
	i = 0
	printProgressBar(0, jumlah_image, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for root, dirs, files in os.walk(input_path):
		for image in files:
			with open(os.path.join(root, image), 'rb') as file:
				src = os.path.join(root, image)
				dst = output_path+"/"+image

				#bagian memproses citra
				obj_detect = combineDetector(src, scaleFactor, minNeighbors)
				x1, y1, x2, y2, img_out = obj_detect.get_face()
				cv2.imwrite(dst, img_out)
				i += 1
				printProgressBar(i, jumlah_image, prefix = 'Progress:', suffix = 'Complete', length = 50)

def preprocessing_face_skin(input_path, output_path):

	if not os.path.exists(output_path):
		os.mkdir(output_path)
		print("Directory "+output_path+" Created!")

	jumlah_image = sum([len(files) for r, d, files in os.walk(input_path)])
	i = 0
	printProgressBar(0, jumlah_image, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for root, dirs, files in os.walk(input_path):
		for image in files:
			with open(os.path.join(root, image), 'rb') as file:
				src = os.path.join(root, image)
				dst = output_path+"/"+image

				#bagian memproses citra
				obj_detect = skinDetector(src)
				x1, y1, x2, y2, img_out = obj_detect.get_face()
				cv2.imwrite(dst, img_out)
				i += 1
				printProgressBar(i, jumlah_image, prefix = 'Progress:', suffix = 'Complete', length = 50)


def preprocessing_face_haar(input_path, output_path, scaleFactor, minNeighbors):

	if not os.path.exists(output_path):
		os.mkdir(output_path)
		print("Directory "+output_path+" Created!")

	jumlah_image = sum([len(files) for r, d, files in os.walk(input_path)])
	i = 0
	printProgressBar(0, jumlah_image, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for root, dirs, files in os.walk(input_path):
		for image in files:
			with open(os.path.join(root, image), 'rb') as file:
				src = os.path.join(root, image)
				dst = output_path+"/"+image

				#bagian memproses citra
				obj_detect = haarDetector(src, scaleFactor, minNeighbors)
				x1, y1, x2, y2, img_out = obj_detect.get_face()
				cv2.imwrite(dst, img_out)
				i += 1
				printProgressBar(i, jumlah_image, prefix = 'Progress:', suffix = 'Complete', length = 50)