import argparse
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from face_cnn import *

def main():
	# ap = argparse.ArgumentParser()
	# ap.add_argument('-i', '--inputFolder', required=True, help='Input path file of dataset')
	# ap.add_argument('-o', '--initialOut', required=True, help='Initial Out')
	# ap.add_argument('-log', '--logFile', required=True, help='log File name')
	# args = ap.parse_args()
	folder_train = "D:/PYTHON_PCD/dataset_test"
	folder_test = "D:/PYTHON_PCD/dataset_train"
	format_f = ".JPG"
	color_img = "rgb"

	checkpoint_path = "data_cnn/cp.ckpt"

	obj_cnn = face_cnn(train = folder_train, test = folder_test, fileFormat= format_f, color=color_img, checkpoint=checkpoint_path)
	data_train, label_train, data_test, label_test, initial = obj_cnn.dataset()
	history = obj_cnn.training(number_epoch=10)

	plt.plot(history.history["accuracy"], label="accuracy_train")
	plt.plot(history.history["val_accuracy"], label="accuracy_test")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.ylim([0.5, 1])
	plt.legend(loc="lower right")
	plt.show()
	# obj_cnn.showModel()
	
	# # Loads the weights
	# model.load_weights(checkpoint_path)

	# # Re-evaluate the model
	# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
	# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

	#https://www.tensorflow.org/tutorials/keras/save_and_load
	#https://www.tensorflow.org/tutorials/keras/classification?hl=id
if __name__ == '__main__':
	main() 