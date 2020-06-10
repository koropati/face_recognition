import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from processing import append_list_as_row, read_csv_string

	

class face_cnn():
	def __init__(self, train, test, fileFormat, color, checkpoint):
		#train folder, test folder, fileFormat (.JPG), color = "grey"
		self.train = train
		self.test =  test
		self.fileFormat = fileFormat
		self.color = color
		self.checkpoint = checkpoint
		self.checkpoint_dir = os.path.dirname(self.checkpoint)
		self.initial_class = self.checkpoint_dir+"/name_class.csv"
		self.data_trains = []
		self.data_tests = []
		self.label_trains = []
		self.label_tests = []

		self.initials = []

		self.model = models.Sequential()
		self.model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)))
		self.model.add(layers.MaxPooling2D((2, 2)))
		self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
		self.model.add(layers.MaxPooling2D((2, 2)))
		self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
		self.model.add(layers.Flatten())
		# model.add(layers.Dense(32, activation='relu'))
		self.model.add(layers.Dense(121, activation='softmax'))

	def showModel(self):
		self.model.summary()

	def dataset(self):
		train_images = []
		train_labels = []
		test_images = []
		test_labels = []
		initial = []
		label = []
		nama_kelas=[]
		urut = 0
		loop = 0
		temp =''
		for i in range(2):
			if i==0:
				source_dir = self.train
			else:
				source_dir = self.test

			for dirpath, dirname, files in os.walk(source_dir):
				loop = 0
				for filename in files:
					if filename.endswith(self.fileFormat):
						filepath = os.path.join(dirpath, filename)
						image = cv2.imread(filepath)
						if self.color == 'grey':
							image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

						data_name = filename.split("-")
						if loop == 0:
							temp = data_name[0]
							urut = 0
							if i==0:
								initial.append(data_name[0])
								nama_kelas = [str(data_name[0])]
								append_list_as_row(self.initial_class,nama_kelas)
						else:
							if temp == data_name[0]:
								temp = data_name[0]
							else:
								temp = data_name[0]
								urut += 1
								if i==0:
									initial.append(data_name[0])
									nama_kelas= [str(data_name[0])]
									append_list_as_row(self.initial_class,nama_kelas)
						if i==0:
							train_images.append(image)
							train_labels.append(urut)
						else:
							test_images.append(image)
							test_labels.append(urut)

					loop += 1

		train_images = np.array(train_images,dtype='float32') #as mnist
		test_images = np.array(test_images,dtype='float32')

		train_labels = np.array(train_labels,dtype='int8') #as mnist
		test_labels = np.array(test_labels,dtype='int8')

		#Normalisasi Data Training dan Test
		train_images = train_images / 255.0
		test_images = test_images / 255.0

		self.data_trains = train_images
		self.data_tests = test_images
		self.label_trains = train_labels
		self.label_tests = test_labels

		self.initials = initial

		print("Done Preprocessing Image !")
		return train_images, train_labels, test_images, test_labels, initial

	def training(self, number_epoch):
		# checkpoint_dir = os.path.dirname(self.checkpoint)
		# Create a callback that saves the model's weights
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint,save_weights_only=True,verbose=1)
		self.model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
		history = self.model.fit(self.data_trains, self.label_trains, epochs=number_epoch, validation_data=(self.data_tests, self.label_tests), callbacks=[cp_callback])
		self.model.save(self.checkpoint_dir+"/model_cnn.h5")
		return history

	def test(self):
		test_loss, test_acc = self.model.evaluate(self.data_tests, self.label_tests)
		print('Test accuracy:', test_acc)
		return test_loss, test_acc


#Membuat Plot atau menampilkan image beserta presentase akurasi berada pada class mana
def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(363), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('green')


class recognize_face(object):
	def __init__(self, img_input, model_path, label_trains):
		self.img_input = img_input
		self.img_input = np.asarray(self.img_input)
		print(self.img_input.shape)
		self.img_input = (np.expand_dims(self.img_input,0))
		print(self.img_input.shape)
		self.myModel = tf.keras.models.load_model(model_path)
		self.label = read_csv_string(label_trains)

	def predict(self):
		prediction = self.myModel.predict(self.img_input)
		label_prediction = np.argmax(prediction[0])
		initial_prediction = self.label[label_prediction]

		return label_prediction, initial_prediction
