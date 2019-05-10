# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray
import random
import ray
import matrixDiv as md
# Importing custom modules

import stochLFSR as lfsr
import stochSobel

#ray.init()

random.seed(20)
lfsrSize = 16
half = 127
auxStr = '{:0'+str(lfsrSize)+'b}'

lfsr_seeds = [0,
			  15,
			  30,
			  60,
			  73,
			  12,
			  6,
			  77,
			  927,
			  71,
			  93,
			  456,
			  888]


r = 5*[0]
# 5 random streams for constants
for i in range (5):
	#r[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))
	r[i] = bitarray(auxStr.format(lfsr_seeds[8]))
# 8 random streams for pixels
rng_z = 8*[0]
for i in range(8):
	#rng_z[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))
	rng_z[i] = bitarray(auxStr.format(lfsr_seeds[0]))

del auxStr

def sobelFilter(img=-1):
	'''Function that calculates Gx and Gy of a 3x3 img in numpy matrix form
	Arguments:
	- img: 3x3 region to process Gx and Gy
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning default (0, 0)")
		return 0, 0
	elif(img.shape!=(3,3)):
		print("Invalid 'img' shape (not 3x3), returning default (0, 0)")
		return 0, 0
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning default (0, 0)")
		return 0, 0
	else:
		z = 8*[0]
		z[0] = img[0][0]
		z[1] = img[0][1]
		z[2] = img[0][2]
		z[3] = img[1][0]
		#no middle pixel
		z[4] = img[1][2]
		z[5] = img[2][0]
		z[6] = img[2][1]
		z[7] = img[2][2]

		result = 0
		for i in range(256):
			s = 8*[0]
			rng_z[0] = lfsr.shift(rng_z[0])
			rng_z[1] = lfsr.shift(rng_z[1])
			rng_z[2] = lfsr.shift(rng_z[2])
			rng_z[3] = lfsr.shift(rng_z[3])
			rng_z[4] = lfsr.shift(rng_z[4])
			rng_z[5] = lfsr.shift(rng_z[5])
			rng_z[6] = lfsr.shift(rng_z[6])
			rng_z[7] = lfsr.shift(rng_z[7])

			s[0] = z[0]>int(rng_z[0].to01()[-8:],2)
			s[1] = z[1]>int(rng_z[1].to01()[-8:],2)
			s[2] = z[2]>int(rng_z[2].to01()[-8:],2)
			s[3] = z[3]>int(rng_z[3].to01()[-8:],2)
			s[4] = z[4]>int(rng_z[4].to01()[-8:],2)
			s[5] = z[5]>int(rng_z[5].to01()[-8:],2)
			s[6] = z[6]>int(rng_z[6].to01()[-8:],2)
			s[7] = z[7]>int(rng_z[7].to01()[-8:],2)

			r0 = lfsr.shift(r[0])
			r1 = lfsr.shift(r[1])
			r2 = lfsr.shift(r[2])
			r3 = lfsr.shift(r[3])
			r4 = lfsr.shift(r[4])

			r0 = half>int(r[0].to01()[-8:],2)
			r1 = half>int(r[1].to01()[-8:],2)
			r2 = half>int(r[2].to01()[-8:],2)
			r3 = half>int(r[3].to01()[-8:],2)
			r4 = half>int(r[4].to01()[-8:],2)

			result = result+stochSobel.sobel(s[0],s[1],s[2],
									s[3],s[4],s[5],s[6],s[7],
									r0,
									r1,
									r2,
									r3,
									r4)
		return result

@ray.remote
def rayCreateEdgeImage(img=-1):
	''' Applies Sobel filter on a NxM image "img" loaded via OpenCV (cv2 package) and
	returns three (N-2)x(M-2) images with sobelX, sobelY and both filters applied.
	2 rows and 2 columns removed to simplify boundary conditions.
	Arguments:
	- img: Region in Grayscale color scheme and np.float64 format
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning empty matrix")
		return np.array([0],np.float64)
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning empty matrix")
		return np.array([0],np.float64)
	else:

		# Create images ignoring last row and column for simplicity in
		# convolution operation
		xy_image = np.zeros([img.shape[0]-2,img.shape[1]-2])
		for i in range(1,img.shape[0]-1):
			for j in range(1,img.shape[1]-1):
				# Get 3x3 submatrixes with np.ix_
				# [i-1,i,i+1] = range(i-1,i+2)
				# kept explicit for clarity
				ixgrid = np.ix_([i-1,i,i+1],[j-1,j,j+1])
				workingArea = img[ixgrid]
				# Call the convolution function
				Gxy = sobelFilter(workingArea)
				xy_image[i-1][j-1] = Gxy

		return xy_image

def rayDetectAndShow(imgpath=0):
	# Load image from path
	# Basic validness check before operating
	if(isinstance(imgpath,str)):
		img = cv.imread(imgpath,cv.IMREAD_GRAYSCALE)
		if(isinstance(img,type(None))):
			print("Image could not be loaded")
			return -1,-1
	else:
		print("Invalid image path")
		return -1,-1

	## Remove this line
	#img = img[np.ix_(range(150,250),range(150,250))]
	img_div = md.div8(img)
	# This is where the processing begins
	ray_ids = 8*[0]
	for i in range(8):
		ray_ids[i] = rayCreateEdgeImage.remote(np.float64(img_div[i]))

	xy_img_part = ray.get(ray_ids)
	first_half = np.hstack((xy_img_part[0],xy_img_part[1],xy_img_part[2],xy_img_part[3]))
	second_half = np.hstack((xy_img_part[4],xy_img_part[5],xy_img_part[6],xy_img_part[7]))
	xy_img = np.vstack((first_half,second_half))

	# Convert back to Grayscale
	xy_img = np.uint8(xy_img)

	# Plot only results
	plt.imshow(xy_img,cmap='gray')
	plt.show()

	return img,xy_img

def createEdgeImage(img=-1):
	''' Applies Sobel filter on a NxM image "img" loaded via OpenCV (cv2 package) and
	returns three (N-2)x(M-2) images with sobelX, sobelY and both filters applied.
	2 rows and 2 columns removed to simplify boundary conditions.
	Arguments:
	- img: Region in Grayscale color scheme and np.float64 format
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning empty matrix")
		return np.array([0],np.float64)
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning empty matrix")
		return np.array([0],np.float64)
	else:
		# Create images ignoring last row and column for simplicity in
		# convolution operation
		xy_image = np.zeros([img.shape[0]-2,img.shape[1]-2])
		for i in range(1,img.shape[0]-1):
			for j in range(1,img.shape[1]-1):
				# Get 3x3 submatrixes with np.ix_
				# [i-1,i,i+1] = range(i-1,i+2)
				# kept explicit for clarity
				ixgrid = np.ix_([i-1,i,i+1],[j-1,j,j+1])
				workingArea = img[ixgrid]
				# Call the convolution function
				Gxy = sobelFilter(workingArea)
				xy_image[i-1][j-1] = Gxy

		return xy_image

def detectAndShow(imgpath=0):
	# Load image from path
	# Basic validness check before operating
	if(isinstance(imgpath,str)):
		img = cv.imread(imgpath,cv.IMREAD_GRAYSCALE)
		if(isinstance(img,type(None))):
			print("Image could not be loaded")
			return -1,-1
	else:
		print("Invalid image path")
		return -1,-1
	# This is where the processing begins
	xy_img = createEdgeImage(np.array(img,np.float64))

	# Convert back to Grayscale
	xy_img = np.uint8(xy_img)

	# Plot side by side for comparison
	#plt.subplot(1,2,1), plt.imshow(img,cmap='gray')
	#plt.title('Original'), plt.xticks([]), plt.yticks([])
	#plt.subplot(1,2,2), plt.imshow(xy_img,cmap='gray')
	#plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

	# Plot only results
	plt.imshow(xy_img,cmap='gray')
	plt.show()

	return img,xy_img

