# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray
from scipy.stats import bernoulli
import ray

# Importing custom modules
import stoch.stochLFSR as lfsr
import stoch.stochSobel as stochSobel

import aux.matrixDiv as md
from aux.auxi import *

# "Environment constants"
lfsrSize = 8
half = 127
auxStr = '{:0'+str(lfsrSize)+'b}'

def verifSobelFilter(z=[],errRate=0.0):
	'''Function that calculates Gx and Gy of a 3x3 img in numpy matrix form
	Arguments:
	- This version gives the same result as sobelFilter, but obsfucates a few operations
	but runs in roughly 1/3 of the time
	- I/O has been modified for functional verification
	'''
	if(errRate<0.0 or errRate>1.0):
		print("Invalid error rate, must be between 0.0 and 1.0")
		return 0
	else:
		half = 127
		# z refers to each pixel in the 3x3 region
		#z = 8*[0]
		## upper row
		#z[0] = img[0][0]
		#z[1] = img[0][1]
		#z[2] = img[0][2]

		## middle row
		#z[3] = img[1][0]
		##no middle pixel
		#z[4] = img[1][2]

		## lower row
		#z[5] = img[2][0]
		#z[6] = img[2][1]
		#z[7] = img[2][2]
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

		result = 0
		
		for i in range(256):
			# Variables for storing next bit of
			# respective stochastic number "s[pixel]"
			s_1 = 8*[0]
			s_2 = 8*[0]
			# Stochastic Number Generation of all constants (0.5 for all)

			r0, r1, r2, r3, r4 = np.greater([half, half, half, half, half],
											[rngSequenceList[0][i],
											 rngSequenceList[1][i],
											 rngSequenceList[2][i],
											 rngSequenceList[3][i],
											 rngSequenceList[4][i]])

			# Stochastic Number Generation of all inputs
			s_1[0], s_1[1], s_2[1], s_1[2], s_1[5], s_1[6], s_2[6], s_1[7], s_2[0], s_1[3], s_2[3], s_2[5], s_2[2], s_1[4], s_2[4], s_2[7] = np.greater([z[0], z[1], z[1], z[2], z[5], z[6], z[6], z[7], z[0], z[3], z[3], z[5], z[2], z[4], z[4], z[7]],
																																						[rngSequenceList[ 5][i],
																																						 rngSequenceList[ 6][i],
																																						 rngSequenceList[ 7][i],
																																						 rngSequenceList[ 8][i],
																																						 rngSequenceList[ 5][i],
																																						 rngSequenceList[ 6][i],
																																						 rngSequenceList[ 7][i],
																																						 rngSequenceList[ 8][i],
																																						 rngSequenceList[ 9][i],
																																						 rngSequenceList[10][i],
																																						 rngSequenceList[11][i],
																																						 rngSequenceList[12][i],
																																						 rngSequenceList[ 9][i],
																																						 rngSequenceList[10][i],
																																						 rngSequenceList[11][i],
																																						 rngSequenceList[12][i]]
																																						)
			

			ans = stochSobel.sobel( s_1[0],
									s_1[1],
									s_1[2],
									s_1[3],
									s_1[4],
									s_1[5],
									s_1[6],
									s_1[7],
									s_2[0],
									s_2[1],
									s_2[2],
									s_2[3],
									s_2[4],
									s_2[5],
									s_2[6],
									s_2[7],
									    r0,
									    r1,
									    r2,
									    r3,
									    r4)

			if(errRate!=0):
				errBit = bernoulli.rvs(errRate,size=1)
				ans = ans^errBit
			result = result+ans 
			
		return result

def altSobelFilter(img=-1,rngSequenceList=[],errRate=0.0):
	'''Function that calculates Gx and Gy of a 3x3 img in numpy matrix form
	Arguments:
	- img: 3x3 region to process Gx and Gy
	- This version gives the same result as sobelFilter, but obsfucates a few operations
	but runs in roughly 1/3 of the time
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning default (0, 0)")
		return 0
	elif(img.shape!=(3,3)):
		print("Invalid 'img' shape (not 3x3), returning default (0, 0)")
		return 0
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning default (0, 0)")
		return 0
	elif(errRate<0.0 or errRate>1.0):
		print("Invalid error rate, must be between 0.0 and 1.0")
		return 0
	else:
		global half
		# z refers to each pixel in the 3x3 region
		z = 8*[0]
		# upper row
		z[0] = img[0][0]
		z[1] = img[0][1]
		z[2] = img[0][2]

		# middle row
		z[3] = img[1][0]
		#no middle pixel
		z[4] = img[1][2]

		# lower row
		z[5] = img[2][0]
		z[6] = img[2][1]
		z[7] = img[2][2]

		result = 0
		
		for i in range(256):
			# Variables for storing next bit of
			# respective stochastic number "s[pixel]"
			s_1 = 8*[0]
			s_2 = 8*[0]
			# Stochastic Number Generation of all constants (0.5 for all)

			r0, r1, r2, r3, r4 = np.greater([half, half, half, half, half],
											[rngSequenceList[0][i],
											 rngSequenceList[1][i],
											 rngSequenceList[2][i],
											 rngSequenceList[3][i],
											 rngSequenceList[4][i]])

			# Stochastic Number Generation of all inputs
			s_1[0], s_1[1], s_2[1], s_1[2], s_1[5], s_1[6], s_2[6], s_1[7], s_2[0], s_1[3], s_2[3], s_2[5], s_2[2], s_1[4], s_2[4], s_2[7] = np.greater([z[0], z[1], z[1], z[2], z[5], z[6], z[6], z[7], z[0], z[3], z[3], z[5], z[2], z[4], z[4], z[7]],
																																						[rngSequenceList[ 5][i],
																																						 rngSequenceList[ 6][i],
																																						 rngSequenceList[ 7][i],
																																						 rngSequenceList[ 8][i],
																																						 rngSequenceList[ 5][i],
																																						 rngSequenceList[ 6][i],
																																						 rngSequenceList[ 7][i],
																																						 rngSequenceList[ 8][i],
																																						 rngSequenceList[ 9][i],
																																						 rngSequenceList[10][i],
																																						 rngSequenceList[11][i],
																																						 rngSequenceList[12][i],
																																						 rngSequenceList[ 9][i],
																																						 rngSequenceList[10][i],
																																						 rngSequenceList[11][i],
																																						 rngSequenceList[12][i]]
																																						)
			

			ans = stochSobel.sobel( s_1[0],
									s_1[1],
									s_1[2],
									s_1[3],
									s_1[4],
									s_1[5],
									s_1[6],
									s_1[7],
									s_2[0],
									s_2[1],
									s_2[2],
									s_2[3],
									s_2[4],
									s_2[5],
									s_2[6],
									s_2[7],
									    r0,
									    r1,
									    r2,
									    r3,
									    r4)

			if(errRate!=0):
				errBit = bernoulli.rvs(errRate,size=1)
				ans = ans^errBit
			result = result+ans 
			
		return result

def sobelFilter(img=-1,errRate=0.0):
	'''Function that calculates Gx and Gy of a 3x3 img in numpy matrix form
	Arguments:
	- img: 3x3 region to process Gx and Gy
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning default (0, 0)")
		return 0
	elif(img.shape!=(3,3)):
		print("Invalid 'img' shape (not 3x3), returning default (0, 0)")
		return 0
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning default (0, 0)")
		return 0
	elif(errRate<0.0 or errRate>1.0):
		print("Invalid error rate, must be between 0.0 and 1.0")
		return 0
	else:
		global half
		global lfsrSize
		global auxStr
		# z refers to each pixel in the 3x3 region
		z = 8*[0]
		# upper row
		z[0] = img[0][0]
		z[1] = img[0][1]
		z[2] = img[0][2]

		# middle row
		z[3] = img[1][0]
		#no middle pixel
		z[4] = img[1][2]

		# lower row
		z[5] = img[2][0]
		z[6] = img[2][1]
		z[7] = img[2][2]

		result = 0

		#5 numbers from 16x16 hadamard matrix that have SCC = 0
		#r_had =[bitarray('1001100110011001'),bitarray('1111000011110000'),
		#		bitarray('1010010110100101'),bitarray('1100001111000011'),
		#		bitarray('1001011010010110')]
		r = [bitarray('10011001'),
			 bitarray('11110000'),
			 bitarray('10100101'),
			 bitarray('11000011'),
			 bitarray('10010110')]

		# 8 random streams for all but center pixel
		# 4 random streams for copies of
		# - z2, z4, z6 and z8 for vertical/horizontal
		# - z1, z3, z7 and z9 for diagonal sobel
		#random.seed(32)
		#rng_z_1 = 8*[0]
		#for i in range(8):
		#	rng_z_1[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))
		# Random values precalculated with the code above:
		rng_z_1 = [bitarray('00010011'),
				   bitarray('11101101'),
				   bitarray('00110110'),
				   bitarray('00100101'),
				   bitarray('01001101'),
				   bitarray('10110010'),
				   bitarray('11100110'),
				   bitarray('00111100')]

		for i in range(256):
			# Variables for storing next bit of
			# respective stochastic number "s[pixel]"
			s_1 = 8*[0]
			s_2 = 8*[0]
			# Stochastic Number Generation of all constants (0.5 for all)
			r0 = half>int(r[0].to01()[-8:],2)
			r1 = half>int(r[1].to01()[-8:],2)
			r2 = half>int(r[2].to01()[-8:],2)
			r3 = half>int(r[3].to01()[-8:],2)
			r4 = half>int(r[4].to01()[-8:],2)

			# Stochastic Number Generation of all inputs
			# Mux 1: (z1 z2 z2_2 z3)
			s_1[0] = z[0]>int(rng_z_1[0].to01()[-8:],2)
			s_1[1] = z[1]>int(rng_z_1[1].to01()[-8:],2)
			s_2[1] = z[1]>int(rng_z_1[2].to01()[-8:],2)
			s_1[2] = z[2]>int(rng_z_1[3].to01()[-8:],2)
			# Mux 2: (z7 z8 z8_2 z9)
			s_1[5] = z[5]>int(rng_z_1[0].to01()[-8:],2)
			s_1[6] = z[6]>int(rng_z_1[1].to01()[-8:],2)
			s_2[6] = z[6]>int(rng_z_1[2].to01()[-8:],2)
			s_1[7] = z[7]>int(rng_z_1[3].to01()[-8:],2)

			# Mux 3: (z1_2 z4_1 z4_2 z7_2)
			s_2[0] = z[0]>int(rng_z_1[4].to01()[-8:],2)
			s_1[3] = z[3]>int(rng_z_1[5].to01()[-8:],2)
			s_2[3] = z[3]>int(rng_z_1[6].to01()[-8:],2)
			s_2[5] = z[5]>int(rng_z_1[7].to01()[-8:],2)
			# Mux 4: (z3_2 z6_1 z6_2 z9_2)
			s_2[2] = z[2]>int(rng_z_1[4].to01()[-8:],2)
			s_1[4] = z[4]>int(rng_z_1[5].to01()[-8:],2)
			s_2[4] = z[4]>int(rng_z_1[6].to01()[-8:],2)
			s_2[7] = z[7]>int(rng_z_1[7].to01()[-8:],2)

			ans = stochSobel.sobel( s_1[0],
									s_1[1],
									s_1[2],
									s_1[3],
									s_1[4],
									s_1[5],
									s_1[6],
									s_1[7],
									s_2[0],
									s_2[1],
									s_2[2],
									s_2[3],
									s_2[4],
									s_2[5],
									s_2[6],
									s_2[7],
									r0,
									r1,
									r2,
									r3,
									r4)

			# Shift all LFSR for constants
			r[0] = bitarray(lfsr.shift(r[0]))
			r[1] = bitarray(lfsr.shift(r[1]))
			r[2] = bitarray(lfsr.shift(r[2]))
			r[3] = bitarray(lfsr.shift(r[3]))
			r[4] = bitarray(lfsr.shift(r[4]))

			# Shift all LFSR for inputs 1
			rng_z_1[0] = bitarray(lfsr.shift(rng_z_1[0]))
			rng_z_1[1] = bitarray(lfsr.shift(rng_z_1[1]))
			rng_z_1[2] = bitarray(lfsr.shift(rng_z_1[2]))
			rng_z_1[3] = bitarray(lfsr.shift(rng_z_1[3]))
			rng_z_1[4] = bitarray(lfsr.shift(rng_z_1[4]))
			rng_z_1[5] = bitarray(lfsr.shift(rng_z_1[5]))
			rng_z_1[6] = bitarray(lfsr.shift(rng_z_1[6]))
			rng_z_1[7] = bitarray(lfsr.shift(rng_z_1[7]))
			
			if(errRate!=0):
				errBit = bernoulli.rvs(errRate,size=1)
				ans = ans^errBit
			result = result+ans 
			
		return result

@ray.remote
def rayCreateEdgeImage(img=-1,errRate=0.0):
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
				Gxy = sobelFilter(workingArea,errRate)
				xy_image[i-1][j-1] = Gxy

		return xy_image

def rayDetectAndShow(imgpath=0,errRate=0.0,show=True):
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
		ray_ids[i] = rayCreateEdgeImage.remote(np.float64(img_div[i]),errRate)

	xy_img_part = ray.get(ray_ids)
	first_half = np.hstack((xy_img_part[0],xy_img_part[1],xy_img_part[2],xy_img_part[3]))
	second_half = np.hstack((xy_img_part[4],xy_img_part[5],xy_img_part[6],xy_img_part[7]))
	xy_img = np.vstack((first_half,second_half))

	# Convert back to Grayscale
	xy_img = np.uint8(xy_img)

	# Plot only results
	if(show):
		plt.imshow(xy_img,cmap='gray')
		plt.show()

	return img,xy_img

@ray.remote
def altRayCreateEdgeImage(img=-1,errRate=0.0):
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
		rngListFile = open('rng_8_bit.txt','r')
		rngSequence = []

		for line in rngListFile:
			rngSequence.append(int(line[:-1],2))
			
		rngSequence = np.array(rngSequence,np.uint8)
		

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
				Gxy = altSobelFilter(workingArea,rngSequenceList,errRate)
				xy_image[i-1][j-1] = Gxy
		rngListFile.close()

		return xy_image

def altRayDetectAndShow(imgpath=0,errRate=0.0,show=True):
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
		ray_ids[i] = altRayCreateEdgeImage.remote(np.float64(img_div[i]),errRate)

	xy_img_part = ray.get(ray_ids)
	first_half = np.hstack((xy_img_part[0],xy_img_part[1],xy_img_part[2],xy_img_part[3]))
	second_half = np.hstack((xy_img_part[4],xy_img_part[5],xy_img_part[6],xy_img_part[7]))
	xy_img = np.vstack((first_half,second_half))

	# Convert back to Grayscale
	xy_img = np.uint8(xy_img)

	# Plot only results
	if(show):
		plt.imshow(xy_img,cmap='gray')
		plt.show()

	return img,xy_img

def createEdgeImage(img=-1,errRate=0.0):
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
				Gxy = sobelFilter(workingArea,errRate)
				xy_image[i-1][j-1] = Gxy

		return xy_image

def altCreateEdgeImage(img=-1,errRate=0.0):
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
		rngListFile = open('rng_8_bit.txt','r')
		rngSequence = []

		for line in rngListFile:
			rngSequence.append(int(line[:-1],2))
			
		rngSequence = np.array(rngSequence,np.uint8)
		

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
				Gxy = altSobelFilter(workingArea,rngSequenceList,errRate)
				xy_image[i-1][j-1] = Gxy
		rngListFile.close()

		return xy_image

def detectAndShow(imgpath=0,errRate=0.0):
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
	xy_img = createEdgeImage(np.array(img,np.float64),errRate)

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
