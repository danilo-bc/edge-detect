# Author: Danilo Cavalcanti
# Importing System Modules
import ray
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray
from scipy.stats import bernoulli
# Importing custom modules
import stoch.stochSobel as stochSobel

# Use the same stochastic filter kernel
from stoch.stochWrapper import sobelFilter

import aux.matrixDiv as md
from aux.iohelpers import *

# "Environment constants"
lfsrSize = 8
half = 127
auxStr = '{:0'+str(lfsrSize)+'b}'

@ray.remote
def ray_createEdgeImage(img=-1,errRate=0.0):
	''' Applies Sobel filter on a NxM image "img" loaded via OpenCV (cv2 package) and
	returns three (N-2)x(M-2) images with both filters applied.
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
		rngSequence = np.array([
		0b10011001,	0b00110010, 0b01100101, 0b11001011, 0b10010111, 0b00101110, 0b01011101, 0b10111010, 0b01110100, 0b11101000, 0b11010001, 0b10100011, 0b01000111, 0b10001110, 0b00011101, 0b00111011, 
		0b01110110, 0b11101100, 0b11011001, 0b10110011, 0b01100111, 0b11001111, 0b10011111, 0b00111110, 0b01111101, 0b11111011, 0b11110110, 0b11101101, 0b11011010, 0b10110100, 0b01101000, 0b11010000, 
		0b10100000, 0b01000000, 0b10000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000001, 0b10000010, 0b00000101, 0b00001011, 0b00010111, 0b00101111, 0b01011110, 0b10111101, 
		0b01111011, 0b11110111, 0b11101110, 0b11011101, 0b10111011, 0b01110111, 0b11101111, 0b11011110, 0b10111100, 0b01111000, 0b11110000, 0b11100001, 0b11000010, 0b10000100, 0b00001001, 0b00010011, 
		0b00100111, 0b01001110, 0b10011101, 0b00111010, 0b01110101, 0b11101011, 0b11010110, 0b10101100, 0b01011000, 0b10110001, 0b01100011, 0b11000111, 0b10001111, 0b00011110, 0b00111100, 0b01111001, 
		0b11110011, 0b11100110, 0b11001101, 0b10011011, 0b00110110, 0b01101101, 0b11011011, 0b10110111, 0b01101111, 0b11011111, 0b10111111, 0b01111111, 0b11111111, 0b11111110, 0b11111101, 0b11111010, 
		0b11110101, 0b11101010, 0b11010101, 0b10101011, 0b01010111, 0b10101110, 0b01011100, 0b10111001, 0b01110011, 0b11100111, 0b11001110, 0b10011100, 0b00111001, 0b01110010, 0b11100100, 0b11001001, 
		0b10010011, 0b00100110, 0b01001101, 0b10011010, 0b00110101, 0b01101010, 0b11010100, 0b10101000, 0b01010000, 0b10100001, 0b01000011, 0b10000110, 0b00001101, 0b00011011, 0b00110111, 0b01101110, 
		0b11011100, 0b10111000, 0b01110000, 0b11100000, 0b11000001, 0b10000011, 0b00000110, 0b00001100, 0b00011000, 0b00110000, 0b01100001, 0b11000011, 0b10000111, 0b00001110, 0b00011100, 0b00111000, 
		0b01110001, 0b11100011, 0b11000110, 0b10001100, 0b00011001, 0b00110011, 0b01100110, 0b11001100, 0b10011000, 0b00110001, 0b01100010, 0b11000100, 0b10001000, 0b00010001, 0b00100011, 0b01000110, 
		0b10001101, 0b00011010, 0b00110100, 0b01101001, 0b11010011, 0b10100111, 0b01001111, 0b10011110, 0b00111101, 0b01111010, 0b11110100, 0b11101001, 0b11010010, 0b10100100, 0b01001000, 0b10010001, 
		0b00100010, 0b01000101, 0b10001010, 0b00010101, 0b00101011, 0b01010110, 0b10101101, 0b01011011, 0b10110110, 0b01101100, 0b11011000, 0b10110000, 0b01100000, 0b11000000, 0b10000000, 0b00000000, 
		0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111110, 0b11111100, 0b11111001, 0b11110010, 0b11100101, 0b11001010, 0b10010100, 0b00101001, 0b01010010, 0b10100101, 
		0b01001011, 0b10010110, 0b00101101, 0b01011010, 0b10110101, 0b01101011, 0b11010111, 0b10101111, 0b01011111, 0b10111110, 0b01111100, 0b11111000, 0b11110001, 0b11100010, 0b11000101, 0b10001011, 
		0b00010110, 0b00101100, 0b01011001, 0b10110010, 0b01100100, 0b11001000, 0b10010000, 0b00100001, 0b01000010, 0b10000101, 0b00001010, 0b00010100, 0b00101000, 0b01010001, 0b10100010, 0b01000100, 
		0b10001001, 0b00010010, 0b00100100, 0b01001001, 0b10010010, 0b00100101, 0b01001010, 0b10010101, 0b00101010, 0b01010101, 0b10101010, 0b01010100, 0b10101001, 0b01010011, 0b10100110, 0b01001100]
		,np.uint8)
		
		#5 numbers from 8x8 hadamard matrix that have SCC = 0
		r = [int(np.where(rngSequence==int('10011001',2))[0]),
			 int(np.where(rngSequence==int('11110000',2))[0]),
			 int(np.where(rngSequence==int('10100101',2))[0]),
			 int(np.where(rngSequence==int('11000011',2))[0]),
			 int(np.where(rngSequence==int('10010110',2))[0])]

		rngSequenceList = []
		for i in range(5):
			temp = rngSequence[r[i]:]
			temp = np.append(temp,rngSequence[0:r[i]])
			rngSequenceList.append(temp)
			 
		# 8 random streams for all but center pixel
		# - z2, z4, z6 and z8 for vertical/horizontal
		# - z1, z3, z7 and z9 for diagonal sobel
		#random.seed(32)
		#rng_z_1 = 8*[0]
		#for i in range(8):
		#	rng_z_1[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))
		# Random values precalculated with the code above:
		rng_z_1 = [int(np.where(rngSequence==int('00010011',2))[0]),
				   int(np.where(rngSequence==int('11101101',2))[0]),
				   int(np.where(rngSequence==int('00110110',2))[0]),
				   int(np.where(rngSequence==int('00100101',2))[0]),
				   int(np.where(rngSequence==int('01001101',2))[0]),
				   int(np.where(rngSequence==int('10110010',2))[0]),
				   int(np.where(rngSequence==int('11100110',2))[0]),
				   int(np.where(rngSequence==int('00111100',2))[0])]
		for i in range(8):
			temp = rngSequence[rng_z_1[i]:]
			temp = np.append(temp,rngSequence[0:rng_z_1[i]])
			rngSequenceList.append(temp)
		
		
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
				Gxy = sobelFilter(workingArea,rngSequenceList,errRate)
				xy_image[i-1][j-1] = Gxy

		return xy_image

def ray_detectAndShow(imgpath=0,errRate=0.0,show=True):
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

	img_div = md.div8(img)
	# This is where the processing begins
	ray_ids = 8*[0]
	for i in range(8):
		ray_ids[i] = ray_createEdgeImage.remote(np.float64(img_div[i]),errRate)

	xy_img_part = ray.get(ray_ids)
	first_half = np.hstack((xy_img_part[0],xy_img_part[1],xy_img_part[2],xy_img_part[3]))
	second_half = np.hstack((xy_img_part[4],xy_img_part[5],xy_img_part[6],xy_img_part[7]))
	xy_img = np.vstack((first_half,second_half))

	# Convert back to Grayscale
	xy_img = np.uint8(xy_img)

	# Plot results
	if(show):
		plt.imshow(xy_img,cmap='gray')
		plt.show()

	return img,xy_img