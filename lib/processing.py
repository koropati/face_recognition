import sys, os, glob, shutil
import cv2, math, xlsxwriter
sys.path.insert(1, 'D:/PYTHON_PCD/lib')
from features import HOG, HAAR
from connection import myDatabase
import string, calendar, datetime, traceback
from csv import writer
import numpy as np
from xml.dom import minidom

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def append_list_as_row(file_name, list_of_elem):
	with open(file_name, 'a+', newline='\n') as write_obj:
		csv_writer = writer(write_obj)
		csv_writer.writerow(list_of_elem)

def read_csv_float(name_file):
	data = np.genfromtxt(name_file,delimiter=',', dtype="float")
	return data

def read_csv_string(name_file):
	data = np.genfromtxt(name_file,delimiter=',', dtype="|U17", autostrip=True)
	return data

def read_csv_int(name_file):
	data = np.genfromtxt(name_file,delimiter=',', dtype="int")
	return data

def eucledian_distance(data_x, data_y):
	distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(data_x, data_y)]))
	return distance

class skinDetector(object):

	#class constructor
	def __init__(self, url):
		self.image = cv2.imread(url)
		if self.image is None:
			print("IMAGE NOT FOUND")
			exit(1)                          
		#self.image = cv2.resize(self.image,(600,600),cv2.INTER_AREA)	
		self.HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
		self.YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
		self.binary_mask_image = self.HSV_image
#================================================================================================================================
	#function to process the image and segment the skin using the HSV and YCbCr colorspaces, followed by the Watershed algorithm
	def get_face(self):
		self.__color_segmentation()
		x_min, y_min, x_max, y_max, img_out = self.__region_based_segmentation()
		img_out = cv2.resize(img_out, (100,100))
		return x_min, y_min, x_max, y_max, img_out

#================================================================================================================================
	#Apply a threshold to an HSV and YCbCr images, the used values were based on current research papers along with some
	# empirical tests and visual evaluation
	def __color_segmentation(self):
		lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
		upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")

		lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
		upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

		#A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
		mask_YCbCr = cv2.inRange(self.YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
		mask_HSV = cv2.inRange(self.HSV_image, lower_HSV_values, upper_HSV_values) 

		self.binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)

#================================================================================================================================
	#Function that applies Watershed and morphological operations on the thresholded image
	def __region_based_segmentation(self):
		x_min = 0
		y_min = 0
		x_max = 0
		y_max = 0
		#morphological operations
		image_foreground = cv2.erode(self.binary_mask_image,None,iterations = 3)     	#remove noise
		dilated_binary_image = cv2.dilate(self.binary_mask_image,None,iterations = 3)   #The background region is reduced a little because of the dilate operation
		ret,image_background = cv2.threshold(dilated_binary_image,1,128,cv2.THRESH_BINARY)  #set all background regions to 128

		image_marker = cv2.add(image_foreground,image_background)   #add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
		image_marker32 = np.int32(image_marker) #convert to 32SC1 format

		cv2.watershed(self.image,image_marker32)
		m = cv2.convertScaleAbs(image_marker32) #convert back to uint8 

		#bitwise of the mask with the input image
		ret,image_mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		output = cv2.bitwise_and(self.image,self.image,mask = image_mask)
		
		#show the images
		# self.show_image(self.image)
		# self.show_image(image_mask)
		# self.show_image(output)
		
		# ret, thresh = cv2.threshold(image_mask, 127, 225, cv2.THRESH_BINARY)
		kernel = np.ones((5,5),np.uint8)
		dilasi = cv2.dilate(image_mask,kernel,iterations = 1)
		erosi = cv2.erode(dilasi,kernel,iterations = 6)
		dilasi2 = cv2.dilate(erosi,kernel,iterations = 2)

		contour, hier = cv2.findContours(dilasi2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contour:
			cv2.drawContours(dilasi2,[cnt],0,255,-1)

		erosi2 = cv2.erode(dilasi2,kernel,iterations = 10) #def = 6
		dilasi3 = cv2.dilate(erosi2,kernel,iterations = 25) #def 25


		contour, hier = cv2.findContours(dilasi3,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		# print(contour)
		data_c = []
		image_src = self.image
		img_src = self.image
		for cnt in contour:
			# x_mentah,y_mentah,w_mentah,h_mentah = cv2.boundingRect(cnt)
			data_c.append(cv2.boundingRect(cnt))
			# img_out_mentah = cv2.rectangle(image_src,(x_mentah,y_mentah),(x_mentah+w_mentah,y_mentah+h_mentah),(0,255,0),2)

		# print(data_c)

		y_terdekat = 0
		limit_kotak = 250*250
		count = 0
		index = 0
		for x, y, w, h in data_c:
			if count == 0:
				y_terdekat = y
			else:
				if (w*h) >= limit_kotak:
					if y_terdekat > y:
						y_terdekat = y
						index = count
					else:
						y_terdekat = y_terdekat

			count += 1

		# print(data_c[index][0])
		x = data_c[index][0]
		y = data_c[index][1]
		w = data_c[index][2]
		h = data_c[index][3]
		# min_sisi = min(w,h) + 150
		min_sisi = w + 150
		# (X,Y), (X+W, Y+H)
		# img_out = cv2.rectangle(self.image,(x,y),(x+min_sisi,y+min_sisi),(0,255,0),2)
		img_out = self.image[y:y+min_sisi, x:x+min_sisi]
		x_min = x
		y_min = y
		x_max = x+min_sisi
		y_max = y+min_sisi
		return x_min, y_min, x_max, y_max, img_out

class haarDetector(object):
	def __init__(self, url, scaleFactor, minNeighbors):
		self.image = cv2.imread(url)
		self.y_center = round(self.image.shape[0]/2)
		self.x_center = round(self.image.shape[1]/2)
		self.url_image = url
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
		self.face_cascade = cv2.CascadeClassifier("D:/PYTHON_PCD/lib/haarcascade_frontalface_default.xml")

	def get_face(self):
		faces=[]
		face_true = []
		x_mins = []
		y_mins = []
		x_maxs = []
		y_maxs = []
		x_min = 0
		y_min = 0
		x_max = 0
		y_max = 0

		x_center_faces = []
		y_center_faces = []

		img_asli = self.image
		img = cv2.cvtColor(img_asli, cv2.COLOR_BGR2GRAY)

		data_wajah = self.face_cascade.detectMultiScale(img, self.scaleFactor, self.minNeighbors, minSize=(700, 700))
		index_true=[]
		if len(data_wajah) > 1:
			#Bagian untuk mencari area wajah yang dekat dengan center citra masukan
			count = 0
			for (x, y, w, h) in data_wajah:
				r = max(w, h) / 2
				centerx = x + w / 2
				centery = y + h / 2
				nx = int(centerx - r)
				ny = int(centery - r)
				nr = int(r * 2)
				faceimg = img_asli[ny:ny+nr, nx:nx+nr]
				x_mins.append(nx)
				y_mins.append(ny) 
				x_maxs.append(nx+nr)
				y_maxs.append(ny+nr)

				x_center_faces.append(nx+(round((nx+nr)-nx)/2))
				y_center_faces.append(ny+(round((ny+nr)-ny)/2))

				faces.append(faceimg)
			
			jarak_dekat = 1000
			index_terdekat = 0
			for x in range(len(data_wajah)):
				jarak_wajah = math.sqrt((x_center_faces[x]-self.x_center)**2+(y_center_faces[x]-self.y_center)**2)
				if jarak_wajah<jarak_dekat:
					jarak_dekat = jarak_wajah
					index_terdekat = x
			img_out = faces[index_terdekat]
			img_out = cv2.resize(img_out, (100,100))
			x_min = x_mins[index_terdekat]
			y_min = y_mins[index_terdekat]
			x_max = x_maxs[index_terdekat]
			y_max = y_maxs[index_terdekat]

					
		elif len(data_wajah) == 1:
			#jika memang di deteksi 1 maka itu yang akan digunakan sebagai citra output
			for (x, y, w, h) in data_wajah:
				r = max(w, h) / 2
				centerx = x + w / 2
				centery = y + h / 2
				nx = int(centerx - r)
				ny = int(centery - r)
				nr = int(r * 2)
				img_out = img_asli[ny:ny+nr, nx:nx+nr]
				img_out = cv2.resize(img_out, (100,100))
				x_min = nx
				y_min = ny
				x_max = nx+nr
				y_max = ny+nr
				
				
		else:
			#jika tidak mampu mendeteksi sama sekali maka ambil area tengah citra, hehehe
			x_min = self.x_center - 500
			y_min = self.y_center - 500
			x_max = self.x_center + 500
			y_max = self.y_center + 500
			img_out = self.image[y_min:y_max, x_min:x_max]
			img_out = cv2.resize(img_out, (100,100))
		
		return x_min, y_min, x_max, y_max, img_out

class combineDetector(object):
	def __init__(self, url, scaleFactor, minNeighbors):
		self.image = cv2.imread(url)
		self.y_center = round(self.image.shape[0]/2)
		self.x_center = round(self.image.shape[1]/2)
		self.url_image = url
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
		self.face_cascade = cv2.CascadeClassifier("D:/PYTHON_PCD/lib/haarcascade_frontalface_default.xml")

	def find_true_face(self, ambigius_img):
		img = cv2.cvtColor(ambigius_img, cv2.COLOR_BGR2GRAY)
		data_wajah = self.face_cascade.detectMultiScale(img, (self.scaleFactor-0.2), self.minNeighbors, minSize=(700, 700))
		if len(data_wajah) > 0 : 
			return True
		return False

	def find_true_face2(self, ambigius_img):
		img = cv2.cvtColor(ambigius_img, cv2.COLOR_BGR2GRAY)
		data_wajah = self.face_cascade.detectMultiScale(img, (self.scaleFactor-0.1), (self.minNeighbors+1), minSize=(700, 700))
		if len(data_wajah) > 0 : 
			return True
		return False

	def get_face(self):
		faces=[]
		face_true = []
		x_mins = []
		y_mins = []
		x_maxs = []
		y_maxs = []
		x_min = 0
		y_min = 0
		x_max = 0
		y_max = 0

		x_center_faces = []
		y_center_faces = []

		img_asli = self.image
		img = cv2.cvtColor(img_asli, cv2.COLOR_BGR2GRAY)

		data_wajah = self.face_cascade.detectMultiScale(img, self.scaleFactor, self.minNeighbors, minSize=(700, 700))
		index_true=[]
		# print("jum wajah : "+str(len(data_wajah)))
		if len(data_wajah) > 1:
			count = 0
			for (x, y, w, h) in data_wajah:

				# cv2.rectangle(img_asli, (x, y), (x+w, y+h), (0, 255, 0), 2)
				r = max(w, h) / 2
				centerx = x + w / 2
				centery = y + h / 2
				nx = int(centerx - r)
				ny = int(centery - r)
				nr = int(r * 2)
				faceimg = img_asli[ny:ny+nr, nx:nx+nr]
				x_mins.append(nx)
				y_mins.append(ny) 
				x_maxs.append(nx+nr)
				y_maxs.append(ny+nr)

				x_center_faces.append(nx+(round((nx+nr)-nx)/2))
				y_center_faces.append(ny+(round((ny+nr)-ny)/2))

				faces.append(faceimg)
				face_true.append(self.find_true_face(faceimg))
				# lastimg = cv2.resize(faceimg, (100, 100))
				# cv2.imwrite(output_name, lastimg)

			index_true = [i for i, val in enumerate(face_true) if val] 
			if len(index_true) == 1:
				# print("data index tunggal")
				for x in index_true:
					img_out = faces[x]
					img_out = cv2.resize(img_out, (100,100))
					x_min = x_mins[x]
					y_min = y_mins[x]
					x_max = x_maxs[x]
					y_max = y_maxs[x]
			elif len(index_true) <= 0 or len(index_true) > 1:
				# img_out = faces[0]
				# x_min = x_mins[0]
				# y_min = y_mins[0]
				# x_max = x_maxs[0]
				# y_max = y_maxs[0]
				jarak_dekat = 1000
				index_terdekat = 0
				# print("Length x_center: "+str(len(x_center_faces)))
				for x in range(len(data_wajah)):
					jarak_wajah = math.sqrt((x_center_faces[x]-self.x_center)**2+(y_center_faces[x]-self.y_center)**2)
					if jarak_wajah<jarak_dekat:
						jarak_dekat = jarak_wajah
						index_terdekat = x
				img_out = faces[index_terdekat]
				img_out = cv2.resize(img_out, (100,100))
				x_min = x_mins[index_terdekat]
				y_min = y_mins[index_terdekat]
				x_max = x_maxs[index_terdekat]
				y_max = y_maxs[index_terdekat]

					
		elif len(data_wajah) == 1:
			for (x, y, w, h) in data_wajah:
				# cv2.rectangle(img_asli, (x, y), (x+w, y+h), (0, 255, 0), 2)
				r = max(w, h) / 2
				centerx = x + w / 2
				centery = y + h / 2
				nx = int(centerx - r)
				ny = int(centery - r)
				nr = int(r * 2)
				img_out = img_asli[ny:ny+nr, nx:nx+nr]
				img_out = cv2.resize(img_out, (100,100))
				x_min = nx
				y_min = ny
				x_max = nx+nr
				y_max = ny+nr
				
				
		else:
			#BAGIAN SKIN DETECTOR
			detector = skinDetector(self.url_image)
			x_min, y_min, x_max, y_max, img_out = detector.get_face()
			
			#END BAGIAN SKIN DETECTOR
		# print(face_true)
		# print(index_true)
		
		return x_min, y_min, x_max, y_max, img_out

def extract_feature_toCSV(input_dir, output_dir, subClass, nama_file):
	#format input : directory without "/"
	#output folder : c:/,,,/.../,,,/...
	if subClass == 'y':
		split_img = '_'
	else:
		split_img = '-'

	name_class = []
	images = glob.glob(input_dir+"/*")
	number = 0
	name = ''
	flag = False
	jum_data = 0
	for url in sorted(images):
		jum_data=jum_data+1
		vect_final = []
		vector = []
		vector2 = []
		identity = url.split('\\')[1]
		nama_image = url.split('\\')[1]
		append_list_as_row(output_dir+'/'+nama_file+'_img_name.csv', [nama_image])
		print("DATA-> "+str(jum_data)+" extract data " + identity)

		identity = identity.split(split_img)[0]
		if flag == False:
			#satu kali di jalankan
			name=identity
			name_class=[name]
			append_list_as_row(output_dir+'/'+nama_file+'_name_class.csv', name_class)
			flag = True
		
		if name != identity:
			name = identity
			number = number + 1
			name_class = [identity]
			append_list_as_row(output_dir+'/'+nama_file+'_name_class.csv', name_class)

		img_in = cv2.imread(url, cv2.IMREAD_GRAYSCALE)

		#HOG FEATURE
		print(identity+" ->Extract HOG")
		hog = HOG(img_in, cell_size=6, bin_size=8) # hasil vector 7200
		vector, image = hog.extract()
		#HAAR FEATURE
		print(identity+" ->Extract HAAR")
		haar = HAAR(img_in, x=3, maxPolling=False, meanPolling=True) #hasil vector 250
		vector2 = haar.extract()

		vect_final.extend(vector)
		vect_final.extend(vector2)
		# vect_final=np.asarray(vect_final) #7200 + 400 = 7600

		append_list_as_row(output_dir+'/'+nama_file+'.csv', vect_final)
		append_list_as_row(output_dir+'/'+nama_file+'_id_class.csv',[number])
		vector = []
		vector2 = []
		vect_final=[]
		nama_image = ''

		print(identity+" ->DONE EXTRACT!")
	print("SELESAI MENG-EXTRACT Feautre!")

def extract_feature_toDB(input_dir, id_feature, subClass, data):
	#format input : directory without "/"
	dataBase_name = ''

	if subClass == 'y':
		split_img = '_'
	else:
		split_img = '-'

	if data == 'train':
		dataBase_name = 'datatrain'
	if data == 'test':
		dataBase_name = 'datatest'
	if data == 'dataset':
		dataBase_name = 'dataset_face'

	obj_DB = myDatabase(host='localhost',user='root',pwd='',dbname=dataBase_name)
	myDB, myCursor = obj_DB.hubungkan()
	
	name_class = []
	images = glob.glob(input_dir+"/*")
	number = 0
	number_sub = 0
	name = ''
	name_sub = ''
	flag = False
	flag_first = False
	flag_first_sub = False
	jum_data = 0
	for url in sorted(images):
		jum_data=jum_data+1
		vect_final = []
		vector = []
		vector2 = []
		# identity = url.split('\\')[1]
		nama_image = url.split('\\')[1]
		print("DATA-> "+str(jum_data)+" extract data " + nama_image)

		#CLASS
		identity = nama_image.split('-')[0] #nama class
		identity2 = nama_image.split('_')[0] #nama SUB class
		if flag == False:
			#satu kali di jalankan
			name=identity
			sql = "INSERT INTO name_class (id_class, name_class) VALUES (%s, %s)"
			val = (number, name)
			myCursor.execute(sql, val)
			myDB.commit()

			name_sub = identity2
			sql = "INSERT INTO name_subclass (id_subclass, name_class) VALUES (%s, %s)"
			val = (number_sub, name_sub)
			myCursor.execute(sql, val)
			myDB.commit()
			flag = True
			flag_first = True
			flag_first_sub = True
		
		if name != identity:
			name = identity
			number = number + 1
			sql = "INSERT INTO name_class (id_class, name_class) VALUES (%s, %s)"
			val = (number, name)
			myCursor.execute(sql, val)
			myDB.commit()
			flag_first = True

		if name_sub != identity2:
			name_sub = identity2
			number_sub = number_sub + 1
			sql = "INSERT INTO name_subclass (id_subclass, name_class) VALUES (%s, %s)"
			val = (number_sub, name_sub)
			myCursor.execute(sql, val)
			myDB.commit()
			flag_first_sub = True
			
		img_in = cv2.imread(url, cv2.IMREAD_GRAYSCALE)
		#HOG FEATURE
		print(identity+" ->Extract HOG")
		hog = HOG(img_in, cell_size=6, bin_size=8) # hasil vector 7200
		vector, image = hog.extract()
		#HAAR FEATURE
		print(identity+" ->Extract HAAR")
		haar = HAAR(img_in, x=3, maxPolling=False, meanPolling=True) #hasil vector 250
		vector2 = haar.extract()

		vect_final.extend(vector)
		vect_final.extend(vector2)
		data_extract = ','.join(map(str, vect_final))

		if data == 'train' or data == 'dataset':
			if flag_first == True:
				sql_centroid = "INSERT INTO data_centroid (id_class, id_subclass, data, image_name, id_feature) VALUES (%s, %s, %s, %s, %s)"
				val_centroid = (number, number_sub, data_extract, nama_image, id_feature)
				myCursor.execute(sql_centroid, val_centroid)
				myDB.commit()
				print("Success Add to centroid")
				flag_first = False

			if flag_first_sub == True:
				sql_centroid = "INSERT INTO data_centroid_subclass (id_class, id_subclass, data, image_name, id_feature) VALUES (%s, %s, %s, %s, %s)"
				val_centroid = (number, number_sub, data_extract, nama_image, id_feature)
				myCursor.execute(sql_centroid, val_centroid)
				myDB.commit()
				print("Success Add to centroid subClass")
				flag_first_sub = False

		
		sql = "INSERT INTO data_feature (id_class, id_subclass, data, image_name, id_feature) VALUES (%s, %s, %s, %s, %s)"
		val = (number, number_sub, data_extract, nama_image, id_feature)
		myCursor.execute(sql, val)
		myDB.commit()
		print(myCursor.rowcount, "Record inserted.")

		data_extract = ''
		vector = []
		vector2 = []
		vect_final=[]
		nama_image = ''
		# features.append(vect_final)

		print(identity+" ->DONE EXTRACT!")
	obj_DB.putuskan()
	print("SELESAI MENG-EXTRACT Feautre!")

def extract_feature(image):
	vect_final = []
	#HOG FEATURE
	hog = HOG(image, cell_size=6, bin_size=8) # hasil vector 7200
	vector, image = hog.extract()
	#HAAR FEATURE
	haar = HAAR(image, x=3, maxPolling=False, meanPolling=True) #hasil vector 250
	vector2 = haar.extract()
	vect_final.extend(vector)
	vect_final.extend(vector2)

	return vect_final

def hitung_knn_batch(tests, trainings, label_test, class_name, label_train, img_name_test, img_name_train, k, out_file):

	eucledian=[]
	eucledians=[]
	label_trains=[]

	idx_k = []

	label_k = []
	labels_k = []

	data_k = []
	datas_k = []

	winner = []

	num = 1
	idx_label_train = 0
	excel_start_row = 1
	jum_akurat = 0
	jum_tdk_akurat = 0
	workbook = xlsxwriter.Workbook(out_file)
	worksheet = workbook.add_worksheet()
	worksheet.write('A1', 'Id test')
	worksheet.write('B1', 'Test Label')
	worksheet.write('C1', 'Id Winner')
	worksheet.write('D1', 'Winner Label')
	worksheet.write('E1', 'AKURAT?')
	index_test = 0
	index_train = 0
	for i in range(tests.shape[0]):
		for j in range(trainings.shape[0]):
			data=eucledian_distance(tests[i], trainings[j])
			print("DATA: " + str(num))
			print("TEST: " + img_name_test[index_test] + " => TRAIN: " + img_name_train[index_train])
			print("EUCLEDIAN = "+str(data))
			eucledian.extend([data])
			num=num+1
			index_train +=1
		index_train = 0
		# label_trains.append(label_train)
		# eucledians.append(eucledian)
		
		append_list_as_row('knn_eucledians.csv', eucledian)

		#find K value eucledian and index
		idx_k = np.argpartition(eucledian, k)
		# print(idx_k)
		# print(label_train)
		label_k = label_train[idx_k[:k]]
		#untuk mencari pemenang tiap test
		# winner.append(np.argmax(np.bincount(label_k)))

		win_label = np.argmax(np.bincount(label_k))
		append_list_as_row('knn_winner.csv', [win_label])
		# print(label_k)
		data_k = np.array(eucledian)[idx_k[:k]]
		# print(data_k)
		akurat = 0
		if label_test[index_test] == win_label:
			akurat = 1
			jum_akurat +=1
		else:
			akurat = 0
			jum_tdk_akurat +=1
		worksheet.write(excel_start_row, 0, label_test[index_test])
		worksheet.write(excel_start_row, 1, img_name_test[index_test])
		worksheet.write(excel_start_row, 2, win_label)
		worksheet.write(excel_start_row, 3, class_name[win_label])
		worksheet.write(excel_start_row, 4, akurat)

		index_test +=1
		excel_start_row +=1

		# labels_k.append(label_k)
		append_list_as_row('knn_labels_k.csv', label_k)
		append_list_as_row('knn_datas_k.csv', data_k)
		# datas_k.append(data_k)
		eucledian=[]

	worksheet.write('A'+str(tests.shape[0]+3), 'DATA')
	worksheet.write('B'+str(tests.shape[0]+3), 'NILAI')
	worksheet.write('C'+str(tests.shape[0]+3), '%')
	worksheet.write(tests.shape[0]+4, 0, 'Akurat')
	worksheet.write(tests.shape[0]+4, 1, jum_akurat)
	worksheet.write(tests.shape[0]+4, 2, (jum_akurat/tests.shape[0])*100)
	worksheet.write(tests.shape[0]+5, 0, 'Tdk Akurat')
	worksheet.write(tests.shape[0]+5, 1, jum_tdk_akurat)
	worksheet.write(tests.shape[0]+5, 2, (jum_tdk_akurat/tests.shape[0])*100)
	worksheet.write(tests.shape[0]+6, 0, 'Nilai K')
	worksheet.write(tests.shape[0]+6, 1, k)
	workbook.close()

def hitung_knn(trainings, class_name, label_train, img_name_train, k, test):
	eucledian=[]
	idx_k = []
	label_k = []
	win_eucledian = []

	index_train = 0
	for j in range(trainings.shape[0]):
		data=eucledian_distance(test, trainings[j])
		print(" => TRAIN: " + img_name_train[index_train])
		print("EUCLEDIAN = "+str(data))
		eucledian.extend([data])
		index_train +=1
	index_train = 0
	# append_list_as_row('hasil_eucledians.csv', eucledian)

	#find K value eucledian and index
	idx_k = np.argpartition(eucledian, k)
	label_k = label_train[idx_k[:k]]

	win_label = np.argmax(np.bincount(label_k))
	win_class = class_name[win_label]
	# print(label_k)
	win_eucledian = np.array(eucledian)[idx_k[:k]]
	# print(data_k)
	return win_label, win_class

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    print("xA: "+str(xA)+" yA: "+str(yA)+" xB: "+str(xB)+" yB: "+str(yB))

    # compute the area of intersection rectangle
    interArea = max(abs(xB - xA), 0) * max(abs(yB - yA), 0)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
	
def hitung_iou(input_path, out_file, metode, out_excel):

	current_folder = ""
	root_dst = out_file
	if not os.path.exists(root_dst):
		os.mkdir(root_dst)
		print("Directory " , root_dst ,  " Created ")

	workbook = xlsxwriter.Workbook(out_excel)
	worksheet = workbook.add_worksheet()
	worksheet.write('A1', 'No')
	worksheet.write('B1', 'Id Image')
	worksheet.write('C1', 'X_min1')
	worksheet.write('D1', 'Y_min1')
	worksheet.write('E1', 'X_max1')
	worksheet.write('F1', 'Y_max1')

	worksheet.write('G1', 'X_min2')
	worksheet.write('H1', 'Y_min2')
	worksheet.write('I1', 'X_max2')
	worksheet.write('J1', 'Y_max2')
	worksheet.write('K1', 'IoU')
	index_excel = 1
	number_image = 1
	for root, dirs, files in os.walk(input_path):
		for xml in files:
			with open(os.path.join(root, xml), 'rb') as file:
				src_xml = os.path.join(root, xml)

				doc = minidom.parse(src_xml)
				url_img = str(doc.getElementsByTagName('path')[0].firstChild.data)
				img_name = str(doc.getElementsByTagName('filename')[0].firstChild.data)
				width_xml = int(doc.getElementsByTagName('width')[0].firstChild.data)
				height_xml = int(doc.getElementsByTagName('height')[0].firstChild.data)
				x_min = int(doc.getElementsByTagName('xmin')[0].firstChild.data)
				y_min = int(doc.getElementsByTagName('ymin')[0].firstChild.data)
				x_max = int(doc.getElementsByTagName('xmax')[0].firstChild.data)
				y_max = int(doc.getElementsByTagName('ymax')[0].firstChild.data)

				citra = cv2.imread(url_img)
				if metode == 'skinDetector':
					detector = skinDetector(url_img)
					xMin, yMin, xMax, yMax, img_predict = detector.get_face()
				elif metode == 'haarDetector':
					detector = combineDetector(url_img, 1.3, 6)
					xMin, yMin, xMax, yMax, img_predict = detector.get_face()
				else:
					detector = haarDetector(url_img, 1.3, 6)
					xMin, yMin, xMax, yMax, img_predict = detector.get_face()

				xMin_gt = height_xml-y_max
				yMin_gt = x_max
				xMax_gt = height_xml-y_min
				yMax_gt = x_min

				boxA = [xMin, yMin, xMax, yMax]
				boxB = [xMin_gt, yMin_gt, xMax_gt, yMax_gt]

				# obj_IoU = iou(boxA, boxB)
				# val_iou = obj_IoU.get()
				val_iou = bb_intersection_over_union(boxA, boxB)
				print(val_iou)
				
				width = height_xml
				height = width_xml

				x_min_rect1 = 20
				y_min_rect1 = height-550
				x_max_rect1 = x_min_rect1+100
				y_max_rect1 = y_min_rect1+100

				x_min_rect2 = 20
				y_min_rect2 = y_max_rect1+120
				x_max_rect2 = x_min_rect2+100
				y_max_rect2 = y_min_rect2+100

				font = cv2.FONT_HERSHEY_SIMPLEX

				worksheet.write(index_excel, 0, number_image)
				worksheet.write(index_excel, 1, img_name)
				worksheet.write(index_excel, 2, xMin)
				worksheet.write(index_excel, 3, yMin)
				worksheet.write(index_excel, 4, xMax)
				worksheet.write(index_excel, 5, yMax)

				worksheet.write(index_excel, 6, xMin_gt)
				worksheet.write(index_excel, 7, yMin_gt)
				worksheet.write(index_excel, 8, xMax_gt)
				worksheet.write(index_excel, 9, yMax_gt)
				worksheet.write(index_excel, 10, val_iou)

				# create box of Ground truth
				img_out = cv2.rectangle(citra, (xMin_gt, yMin_gt), (xMax_gt, yMax_gt), (0,255,0), 4)
				img_out = cv2.rectangle(img_out, (x_min_rect1, y_min_rect1), (x_max_rect1, y_max_rect1), (0,255,0), cv2.FILLED)
				img_out = cv2.putText(img_out, 'Ground Truth', (x_max_rect1+80, y_max_rect1), font, 4, (0,255,0), 10)
				# Create box of Predection
				img_out = cv2.rectangle(img_out, (xMin, yMin), (xMax, yMax), (255,0,0), 4)
				img_out = cv2.rectangle(img_out, (x_min_rect2, y_min_rect2), (x_max_rect2, y_max_rect2), (255,0,0), cv2.FILLED)
				img_out = cv2.putText(img_out, 'Prediction', (x_max_rect2+80, y_max_rect2), font, 4, (255,0,0), 10)

				img_out = cv2.putText(img_out, "IoU: {:.4f}".format(val_iou), (20, y_max_rect2+200), cv2.FONT_HERSHEY_SIMPLEX, 4, (221, 0, 255), 10)

				dst = root_dst +"/"+ img_name
				cv2.imwrite(dst, img_out)
				print(dst)
				index_excel += 1
				number_image += 1

	workbook.close()
	print("Selesai Memprocessing Images!")
	sys.exit()

def split_img(src_dir, dst_train, dst_test):
	for root, dirs, files in os.walk(src_dir):
		for image in files:
			with open(os.path.join(root, image), 'rb') as file:
				current_folder = root.split("\\")[-1]
				src = os.path.join(root, image)
				num_img = image.split("_")[-1]
				id_img = int(num_img.split(".")[0])
				
				if (id_img > 0 and id_img < 8) or (id_img > 10 and id_img < 18) or (id_img > 20 and id_img < 28):
					shutil.copyfile(src, dst_train+'/'+image)
					print(dst_train+'/'+image)
				else:
					shutil.copyfile(src, dst_test+'/'+image)
					print(dst_test+'/'+image)