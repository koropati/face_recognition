import argparse
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import hitung_iou
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputXml', required=True, help='Input Folder Xml')
	ap.add_argument('-o', '--outFolder', required=True, help='Output Folder initial')
	ap.add_argument('-m', '--metode', required=True, help='Metode detector (combine/skinDetector')
	ap.add_argument('-excel', '--outExcel', required=True, help='output excel file')
	args = ap.parse_args()

	hitung_iou(args.inputXml, args.outFolder, args.metode, args.outExcel)

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python hitung_iou.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/HASIL_LABEL_TEST" -o "D:/PYTHON_PCD/Image/Hasil_IoU_Combine" -m "combineDetector" -excel "D:/PYTHON_PCD/DATA_IOU_Combine.xlsx"
	# 
	# python hitung_iou.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/HASIL_LABEL_TEST" -o "D:/PYTHON_PCD/Image/Hasil_IoU_Skin" -m "skinDetector" -excel "D:/PYTHON_PCD/DATA_IOU_Skin.xlsx"
	# 
	# python hitung_iou.py -i "D:/PROJECT_SKRIPSI/PROGRAM_CORE_PCD/HASIL_LABEL_TEST" -o "D:/PYTHON_PCD/Image/Hasil_IoU_Haar" -m "haarDetector" -excel "D:/PYTHON_PCD/DATA_IOU_Haar.xlsx"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()