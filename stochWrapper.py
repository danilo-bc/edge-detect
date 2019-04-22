# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray
import random
# Importing auxiliary modules

from stochLFSR import *
from stochSobel import *

random.seed(20)
r0 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
r1 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
r2 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
r3 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
r4 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))

rng_z1 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z2 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z3 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z4 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z6 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z7 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z8 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))
rng_z9 = lfsr(8,'{0:08b}'.format(random.getrandbits(8)))


sSobel = stochSobel()

def SNG(det,rng):
    return det>int(rng,2)

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
		z1 = img[0][0]
		z2 = img[0][1]
		z3 = img[0][2]
		z4 = img[1][0]
		#z5 suppressed
		z6 = img[1][2]
		z7 = img[2][0]
		z8 = img[2][1]
		z9 = img[2][2]
        
		result = 0
		for i in range(256):
			s1 = SNG(z1,rng_z1.shift())
			s2 = SNG(z2,rng_z2.shift())
			s3 = SNG(z3,rng_z3.shift())
			s4 = SNG(z4,rng_z4.shift())
			s6 = SNG(z6,rng_z6.shift())
			s7 = SNG(z7,rng_z7.shift())
			s8 = SNG(z8,rng_z8.shift())
			s9 = SNG(z9,rng_z9.shift())
			
			result = result+sSobel.sobel(s1,s2,s3,
									s4,s6,s7,s8,s9,
									r0.next(),
									r1.next(),
									r2.next(),
									r3.next(),
									r4.next())
		return result

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
		img = img[np.ix_(range(0,60),range(0,60))]
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
