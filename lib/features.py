import cv2
import numpy as np
import math
import skimage.measure

#======================================== HOG FEATURE ==============================================
class HOG(object):
    def __init__(self, image, cell_size, bin_size):
        self.img = image
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        self.img = np.asarray(self.img)
        self.img = np.sqrt(self.img / float(np.max(self.img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) != int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.extend(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

#======================================== END HOG FEATURE ==============================================
#
#======================================== LBP FEATURES =================================================

class LBP(object):
    def __init__(self, image):
        self.img = image
        # self.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        self.img = np.asarray(self.img)

    def get_pixel(self, img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def lbp_calculated_pixel(self, img, x, y):
        center = img[x][y]
        val_ar = []
        val_ar.append(self.get_pixel(img, center, x-1, y+1))     # top_right
        val_ar.append(self.get_pixel(img, center, x, y+1))       # right
        val_ar.append(self.get_pixel(img, center, x+1, y+1))     # bottom_right
        val_ar.append(self.get_pixel(img, center, x+1, y))       # bottom
        val_ar.append(self.get_pixel(img, center, x+1, y-1))     # bottom_left
        val_ar.append(self.get_pixel(img, center, x, y-1))       # left
        val_ar.append(self.get_pixel(img, center, x-1, y-1))     # top_left
        val_ar.append(self.get_pixel(img, center, x-1, y))       # top

        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val

    def reshape_row(self, data_arr):
        row, col = data_arr.shape
        data_out = np.reshape(data_arr, row, order='F')
        return data_out

    def crop_square_4(self, data_array):
        img = data_array
        height, width = img.shape
        img1 = img[0:int(height/2), 0:int(width/2)]
        img2 = img[0:int(height/2), int(width/2):width]
        img3 = img[int(height/2):height, 0:int(width/2)]
        img4 = img[int(height/2):height, int(width/2):width]

        return img1, img2, img3, img4


    def extract(self):
        image_gray = self.img
        height, width = image_gray.shape

        img_lbp = np.zeros((height, width), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = self.lbp_calculated_pixel(image_gray, i, j)

        img1, img2, img3, img4 = self.crop_square_4(img_lbp)

        hist_img1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist_img2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist_img3 = cv2.calcHist([img3], [0], None, [256], [0, 256])
        hist_img4 = cv2.calcHist([img4], [0], None, [256], [0, 256])

        x1 = np.asarray(self.reshape_row(hist_img1/sum(sum(hist_img1))))
        x2 = np.asarray(self.reshape_row(hist_img2/sum(sum(hist_img2))))
        x3 = np.asarray(self.reshape_row(hist_img3/sum(sum(hist_img3))))
        x4 = np.asarray(self.reshape_row(hist_img4/sum(sum(hist_img4))))
        feature_vector = []
        feature_vector.extend(x1)
        feature_vector.extend(x2)
        feature_vector.extend(x3)
        feature_vector.extend(x4)

        return feature_vector, img_lbp

#======================================== END LBP FEATURE ==============================================
#
#======================================== HAAR FEATURE =================================================
#Feature Haar
class HAAR(object):
    def __init__(self, image, x, maxPolling, meanPolling):
        self.img = image
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        self.img = np.asarray(self.img)
        self.x = x
        self.maxPolling = maxPolling
        self.meanPolling = meanPolling

    def hitung_perkalian(self, data1, data2): 
        x1 = data1.shape[0]
        y1 = data1.shape[1]
        x2 = data2.shape[0]
        y2 = data2.shape[1]
        # hasil = [[0 for x in range(x1)] for y in range(y1)] 
        temp=[]
        hasil=[]

        if x1 == x2 and y1 == y2:
            for i in range(x1):
                for j in range(y1):
                    temp.extend([data1[i][j]*data2[i][j]])
                hasil.append(temp)
                temp=[]
            hasil=np.asarray(hasil)

        else:
            print("Error: Ukuran Tak sama")
        return hasil

    def extract(self):
        feature = []
        haar1 = np.array([[-1,-1,self.x], [-1,-1,self.x], [-1,-1,self.x]])
        haar2 = np.array([[self.x,self.x,self.x], [-1,-1,-1], [-1,-1,-1]])
        haar3 = np.array([[-1,self.x,-1], [-1,self.x,-1], [-1,self.x,-1]])
        haar4 = np.array([[self.x,-1,-1], [self.x,-1,-1], [self.x,-1,-1]])
        temp = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        rows,cols = self.img.shape[:2]

        out_1=[[0 for x in range(rows)] for y in range(cols)] 
        out_2=[[0 for x in range(rows)] for y in range(cols)] 
        out_3=[[0 for x in range(rows)] for y in range(cols)] 
        out_4=[[0 for x in range(rows)] for y in range(cols)]
        out_5=[[0 for x in range(rows)] for y in range(cols)] 

        for i in range(rows):
            for j in range(cols):
                # print("I = ", i, " J = ", j)
                if (j == 0) and (i == 0):
                    # print("IF #1")
                    temp[1][1]=self.img[i][j]
                    temp[1][2]=self.img[i][j+1]
                    temp[2][1]=self.img[i+1][j]
                    temp[2][2]=self.img[i+1][j+1]
                elif (j>0 and j<(cols-1)) and (i == 0) :
                    # print("IF #2")
                    temp[1][0]=self.img[i][j-1]
                    temp[1][1]=self.img[i][j]
                    temp[1][2]=self.img[i][j+1]
                    temp[2][0]=self.img[i+1][j-1]
                    temp[2][1]=self.img[i+1][j]
                    temp[2][2]=self.img[i+1][j+1]
                elif (j == (cols-1)) and (i == 0) :
                    # print("IF #3")
                    temp[1][0]=self.img[i][j-1]
                    temp[1][1]=self.img[i][j]
                    temp[2][0]=self.img[i+1][j-1]
                    temp[2][1]=self.img[i+1][j]
                elif (j == 0) and (i > 0 and i < (rows-1)):
                    # print("IF #4")
                    temp[0][1] = self.img[i-1][j]
                    temp[0][2] = self.img[i-1][j+1]
                    temp[1][1] = self.img[i][j]
                    temp[1][2] = self.img[i][j+1]
                    temp[2][1] = self.img[i+1][j]
                    temp[2][2] = self.img[i+1][j+1]
                elif (j == (cols-1)) and (i > 0 and i < (rows-1)):
                    # print("IF #5")
                    temp[0][0] = self.img[i-1][j-1]
                    temp[0][1] = self.img[i-1][j]
                    temp[1][0] = self.img[i][j-1]
                    temp[1][1] = self.img[i][j]
                    temp[2][0] = self.img[i+1][j-1]
                    temp[2][1] = self.img[i+1][j]
                elif (j == 0) and (i == (rows-1)):
                    # print("IF #6")
                    temp[0][1] = self.img[i-1][j]
                    temp[0][2] = self.img[i-1][j+1]
                    temp[1][1] = self.img[i][j]
                    temp[1][2] = self.img[i][j+1]
                elif (j > 0 and j < (cols-1)) and (i == (rows-1)):
                    # print("IF #7")
                    temp[0][0] = self.img[i-1][j-1]
                    temp[0][1] = self.img[i-1][j]
                    temp[0][2] = self.img[i-1][j+1]
                    temp[1][0] = self.img[i][j-1]
                    temp[1][1] = self.img[i][j]
                    temp[1][2] = self.img[i][j+1]
                elif (j == (cols-1)) and (i == (rows-1)):
                    #POJOK KANAN BAWAH
                    # print("IF #8")
                    temp[0][0] = self.img[i-1][j-1]
                    temp[0][1] = self.img[i-1][j]
                    temp[1][0] = self.img[i][j-1]
                    temp[1][1] = self.img[i][j]
                else:
                    # print("IF #9")
                    temp[0][0] = self.img[i-1][j-1]
                    temp[0][1] = self.img[i-1][j]
                    temp[0][2] = self.img[i-1][j+1]
                    temp[1][0] = self.img[i][j-1]
                    temp[1][1] = self.img[i][j]
                    temp[1][2] = self.img[i][j+1]
                    temp[2][0] = self.img[i+1][j-1]
                    temp[2][1] = self.img[i+1][j]
                    temp[2][2] = self.img[i+1][j+1]

                # print("TEMP - >",temp.shape)
                # print("HAAR - >",haar1.shape)
                # print(temp)
                
                a = np.sum(self.hitung_perkalian(temp, haar1))
                b = np.sum(self.hitung_perkalian(temp, haar2))
                c = np.sum(self.hitung_perkalian(temp, haar3))
                d = np.sum(self.hitung_perkalian(temp, haar4))

                out_1[i][j] = a
                out_2[i][j] = b
                out_3[i][j] = c
                out_4[i][j] = d
                out_5[i][j] = math.sqrt((a**2) + (b**2) + (c**2) + (d**2))

                temp = temp * 0
                
        if self.meanPolling == True:
            out_1=skimage.measure.block_reduce(np.asarray(out_1), (4,4), np.mean)
            out_2=skimage.measure.block_reduce(np.asarray(out_2), (4,4), np.mean)
            out_3=skimage.measure.block_reduce(np.asarray(out_3), (4,4), np.mean)
            out_4=skimage.measure.block_reduce(np.asarray(out_4), (4,4), np.mean)
            out_5=skimage.measure.block_reduce(np.asarray(out_5), (4,4), np.mean)

        if self.maxPolling == True:
            out_1=skimage.measure.block_reduce(np.asarray(out_1), (4,4), np.max)
            out_2=skimage.measure.block_reduce(np.asarray(out_2), (4,4), np.max)
            out_3=skimage.measure.block_reduce(np.asarray(out_3), (4,4), np.max)
            out_4=skimage.measure.block_reduce(np.asarray(out_4), (4,4), np.max)
            out_5=skimage.measure.block_reduce(np.asarray(out_5), (4,4), np.max)

        out_data1=np.asarray(out_1).sum(axis=0)
        out_data2=np.asarray(out_2).sum(axis=0)
        out_data3=np.asarray(out_3).sum(axis=0)
        out_data4=np.asarray(out_4).sum(axis=0)
        out_data5=np.asarray(out_5).sum(axis=0)

        out_data6=np.asarray(out_1).sum(axis=1)
        out_data7=np.asarray(out_2).sum(axis=1)
        out_data8=np.asarray(out_3).sum(axis=1)
        out_data9=np.asarray(out_4).sum(axis=1)
        out_data10=np.asarray(out_5).sum(axis=1)

        total1 = np.sum(out_data1)
        total2 = np.sum(out_data2)
        total3 = np.sum(out_data3)
        total4 = np.sum(out_data4)
        total5 = np.sum(out_data5)

        total6 = np.sum(out_data6)
        total7 = np.sum(out_data7)
        total8 = np.sum(out_data8)
        total9 = np.sum(out_data9)
        total10 = np.sum(out_data10)

        feature.extend(out_data1/total1)
        feature.extend(out_data2/total2)
        feature.extend(out_data3/total3)
        feature.extend(out_data4/total4)
        feature.extend(out_data5/total5)

        feature.extend(out_data6/total6)
        feature.extend(out_data7/total7)
        feature.extend(out_data8/total8)
        feature.extend(out_data9/total9)
        feature.extend(out_data10/total10)

        # feature = np.asarray(feature)
        return feature