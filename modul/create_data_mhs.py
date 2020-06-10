import xlsxwriter
import xlrd
import sys
import argparse
import datetime, random, string

def get_random_alphaNumeric_string(stringLength=8):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))


def create_data_mhs(input_file, output_file):
	file_input = xlrd.open_workbook(input_file)
	file_input_sheet = file_input.sheet_by_index(0)
	jumlah_data = file_input_sheet.nrows
	email=''
	name=''
	password=''
	nim=''
	tgl_lahir = ''
	name_valid = []
	avatar='images/user_profile/default.png'

	workbook = xlsxwriter.Workbook(output_file)
	worksheet = workbook.add_worksheet()
	
	for x in range(jumlah_data):
		name = file_input_sheet.cell_value(x,0)
		nim = str(int(file_input_sheet.cell_value(x,1)))
		tgl_lahir = str(datetime.datetime(*xlrd.xldate_as_tuple(file_input_sheet.cell_value(x,3),file_input.datemode)))

		nama_lengkap = name.split(" ")
		for kata in nama_lengkap:
			if len(kata) > 2:
				if '.' not in kata:
					name_valid.extend([kata])
		email='.'.join(name_valid)+'@absen.com'
		email=email.lower()

		tgl_aja=tgl_lahir.split(" ")[0]
		tgl=tgl_aja.split("-")
		password=nim+'@'+''.join(tgl)

		worksheet.write(x, 0, name)
		worksheet.write(x, 1, email)
		worksheet.write(x, 2, password)
		worksheet.write(x, 3, nim)
		worksheet.write(x, 4, avatar)

		name_valid=[]
		nama_lengkap=''
		name=''
		nim=''
		email=''
		tgl=''
		password=''

	workbook.close()


def generate_user(input_file, output_file):
	file_input = xlrd.open_workbook(input_file)
	file_input_sheet = file_input.sheet_by_index(0)
	jumlah_data = file_input_sheet.nrows
	email=''
	name=''
	password=''
	nim=''
	name_valid = []
	avatar='images/user_profile/default.png'

	workbook = xlsxwriter.Workbook(output_file)
	worksheet = workbook.add_worksheet()
	
	for x in range(jumlah_data):
		name = file_input_sheet.cell_value(x,0)
		nim = str(int(file_input_sheet.cell_value(x,1)))
		# tgl_lahir = str(datetime.datetime(*xlrd.xldate_as_tuple(file_input_sheet.cell_value(x,3),file_input.datemode)))

		email=nim+'@absen.com'
		email=email.lower()

		password=get_random_alphaNumeric_string(8).lower()

		worksheet.write(x, 0, name)
		worksheet.write(x, 1, email)
		worksheet.write(x, 2, password)
		worksheet.write(x, 3, nim)
		worksheet.write(x, 4, avatar)

		name_valid=[]
		nama_lengkap=''
		name=''
		nim=''
		email=''
		tgl=''
		password=''

	workbook.close()


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--inputFile', required=True, help='Input File Excel')
	ap.add_argument('-o', '--outputFile', required=True, help='Output File Excel')
	args = ap.parse_args()

	generate_user(args.inputFile, args.outputFile)
	print("Selesai!")

	######################################## RUN PROGRAM BY TYPE ################################################
	#
	# python create_data_mhs.py -i "D:/PYTHON_PCD/data_user_mhs.xlsx" -o "D:/PYTHON_PCD/data_user_mhs_password.xlsx"
	# 
	#############################################################################################################

if __name__ == '__main__':
	main()