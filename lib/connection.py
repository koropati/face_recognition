import mysql.connector
import numpy as np
from mysql.connector import Error

class myDatabase(object):
	def __init__(self, host, user, pwd, dbname):
		self.host = host
		self.user = user
		self.pwd = pwd
		self.dbname = dbname

		try:
			self.connection = mysql.connector.connect(
				host=host,
				database=dbname,
				user=user,
				password=pwd)

			if self.connection.is_connected():
				db_Info = self.connection.get_server_info()
				print("Connected to MySQL Server version ", db_Info)
				self.cursor = self.connection.cursor()
				self.cursor.execute("select database();")
				record = self.cursor.fetchone()
				print("You're connected to database: ", record)

		except Error as e:
			print("Error while connecting to MySQL", e)

	def hubungkan(self):
		return self.connection, self.cursor

	def putuskan(self):
		if (self.connection.is_connected()):
			self.cursor.close()
			self.connection.close()
			print("MySQL connection is closed")
		else:
			print("Something wrong!, when closing connection")

	def ALL_DATA_FEATURE(self, tabel):
		tabel_selected = tabel
		self.cursor.execute("SELECT data FROM "+tabel_selected+" WHERE id_feature = 0")
		result = self.cursor.fetchall() #return a list
		data_out = []
		for x in result:
		  data_out.extend([j.split(",") for j in x])

		arr = np.array(data_out)
		float_arr = arr.astype(np.float)

		self.cursor.execute("SELECT id_class FROM "+tabel_selected+" WHERE id_feature = 0")
		id_class = self.cursor.fetchall()
		class_id = np.array(id_class)

		self.cursor.execute("SELECT id_subclass FROM "+tabel_selected+" WHERE id_feature = 0")
		id_subclass = self.cursor.fetchall()
		subclass_id = np.array(id_subclass)

		return class_id, subclass_id, float_arr

	def GET_NAME_CLASS(self, id_class):
		self.cursor.execute("SELECT name_class FROM name_class WHERE id_class = "+str(id_class))
		result = self.cursor.fetchone()
		return result

	def UPLOAD_DATA_KMEANS(self, id_class_kmeans, data_kmeans, id_feature):
		data = ','.join(map(str, data_kmeans))
		sql = "INSERT INTO data_feature_kmeans (id_class, data, id_feature) VALUES (%s, %s, %s)"
		val = (id_class_kmeans, data, id_feature)
		self.cursor.execute(sql, val)
		self.connection.commit()
		# print(self.cursor.rowcount, "record inserted.")
	
	def READ_DATA_KMEANS(self):
		self.cursor.execute("SELECT data FROM data_feature_kmeans WHERE id_feature = 0")
		result = self.cursor.fetchall() #return a list
		data_out = []
		for x in result:
		  data_out.extend([j.split(",") for j in x])

		arr = np.array(data_out)
		float_arr = arr.astype(np.float)

		self.cursor.execute("SELECT id_class FROM data_feature_kmeans WHERE id_feature = 0")
		id_class = self.cursor.fetchall()
		class_id = np.array(id_class)

		return float_arr, class_id