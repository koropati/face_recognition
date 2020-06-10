import cv2
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import combineDetector, extract_feature, hitung_knn
from processing import read_csv_string, read_csv_float, read_csv_int
import numpy as np

def main():
	url='1915051051-depan_8.JPG'
	img_asli = cv2.imread(url)
	obj_detect = combineDetector(url=url, scaleFactor=1.3, minNeighbors=6)
	x1, y1, x2, y2, face_image = obj_detect.get_face()
	img_face = cv2.resize(face_image, (100,100))
	gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)

	vector_feature = extract_feature(gray)
	print(len(vector_feature))

	print("MELOAD data TRAIN ..")
	data_train = read_csv_float('D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/Hasil_KNN/INDIVIDU_121_CLASS/data_train.csv')
	print("MELOAD data LABEL TRAIN ..")
	label_train = read_csv_int('D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/Hasil_KNN/INDIVIDU_121_CLASS/data_train_id_class.csv')
	print("MELOAD data NAMA CLASS ..")
	class_name = read_csv_string('D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/Hasil_KNN/INDIVIDU_121_CLASS/data_test_name_class.csv')
	print("MELOAD data nama Image Train...")
	img_name_train = read_csv_string('D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/Hasil_KNN/INDIVIDU_121_CLASS/data_train_img_name.csv')

	win_label, win_class = hitung_knn(data_train, class_name, label_train, img_name_train, 5, vector_feature)
	print("Terdeteksi Sebagai mahasiswa NIM: "+win_class)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.rectangle(img_asli, (x1, y1), (x2, y2), (0,255,0), 4)
	cv2.putText(img_asli, win_class, (x1, (y2+200)), font, 4, (0,255,0), 18)
	cv2.imwrite('hasil_klasifikasi.JPG', img_asli)

if __name__ == '__main__':
	main()