import mysql.connector
import numpy as np
from itertools import chain
mydb = mysql.connector.connect(
	host = "localhost",
	user = "root",
	passwd = "",
	# database = "dataset_face"
	# database = "datatest"
	database = "datatrain"
)

mycursor = mydb.cursor()
#mycursor.execute("CREATE DATABASE dataset_face")
# mycursor.execute("SHOW DATABASES")
# for x in mycursor:
# 	print(x)

##### UNTUK DATABASE TRAIN ############
#mycursor.execute("CREATE TABLE name_feature (id INT AUTO_INCREMENT PRIMARY KEY, id_feature INT(4), name_feature VARCHAR(255))")
#mycursor.execute("CREATE TABLE name_class (id INT AUTO_INCREMENT PRIMARY KEY, id_class INT(8), name_class VARCHAR(255))")
#mycursor.execute("CREATE TABLE name_subclass (id INT AUTO_INCREMENT PRIMARY KEY, id_subclass INT(8), name_class VARCHAR(255))")
#mycursor.execute("CREATE TABLE data_feature (id INT AUTO_INCREMENT PRIMARY KEY, id_class INT(8), id_subclass INT(8),data LONGTEXT, image_name VARCHAR(255), id_feature INT(4))")
#mycursor.execute("CREATE TABLE data_centroid (id INT AUTO_INCREMENT PRIMARY KEY, id_class INT(8), id_subclass INT(8), data LONGTEXT, image_name VARCHAR(255), id_feature INT(4))")
#mycursor.execute("CREATE TABLE data_centroid_subclass (id INT AUTO_INCREMENT PRIMARY KEY, id_class INT(8), id_subclass INT(8),data LONGTEXT, image_name VARCHAR(255), id_feature INT(4))")
# mycursor.execute("SHOW TABLES")

mycursor.execute("CREATE TABLE data_feature_kmeans (id INT AUTO_INCREMENT PRIMARY KEY, id_class INT(8),data LONGTEXT, id_feature INT(4))")

# for x in mycursor:
#   print(x) 

# mycursor.execute("ALTER TABLE data_feature ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY")
# 

#### BAGIAN INSERT DATABASE
# data_in = [1, 2, 4, 4, 3, 3, 3, 6, 5]
# print(data_in)
# var_string = ','.join(map(str, data_in))
# sql = "INSERT INTO data_feature (id_class, data) VALUES (%s, %s)"
# val = (2, var_string)
# mycursor.execute(sql, val)
# mydb.commit()
# print(mycursor.rowcount, "record inserted.")


#### BAGIAN SELECT DATABASE
# sql = "SELECT data FROM data_feature WHERE id_feature = 0"
# mycursor.execute(sql)
# myresult = mycursor.fetchall()

# print(myresult)


# data_out = []

# for x in myresult:
#   data_out.extend([j.split(",") for j in x])


# arr = np.asarray(data_out)
# float_arr = arr.astype(np.float)
# print(type(float_arr))
# print(float_arr.shape)
# print(float_arr[0][0])

# sql = "SELECT MAX(id_class) FROM data_feature WHERE id_feature = 0"
# mycursor.execute(sql)
# jumlah_class = mycursor.fetchone()
# print(jumlah_class[0])
# 
# 


# sql = "SELECT id_class, data FROM data_feature WHERE id_feature = 0"
# mycursor.execute(sql)
# myresult = mycursor.fetchall()

# data=[]
# for x in myresult:
# 	data.append(list(chain.from_iterable(x)))
# # data = list(chain.from_iterable(myresult))
# print(len(data))
