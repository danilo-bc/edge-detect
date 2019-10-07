# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray
from scipy.stats import bernoulli
import random
import ray

# Importing custom modules
import matrixDiv as md
import stochLFSR as lfsr
import stochSobel

# "Environment constants"
random.seed(32)
lfsrSize = 8
half = 127
auxStr = '{:0'+str(lfsrSize)+'b}'


def sobelFilter(img=-1,r=[],rng_z_1=[],errRate=0.0):
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
		for i in range(255):
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
			s_1[5] = z[6]>int(rng_z_1[0].to01()[-8:],2)
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
		global lfsrSize
		global lfsr_seeds
		global auxStr
		r = 5*[0]
		root_lfsr = bitarray('1010010110100101')
		#5 numbers from 16x16 hadamard matrix that have SCC = 0
		#r_had =[bitarray('1001100110011001'),bitarray('1111000011110000'),
		#		bitarray('1010010110100101'),bitarray('1100001111000011'),
		#		bitarray('1001011010010110')]
		r_had =[bitarray('10011001'),bitarray('11110000'),
				bitarray('10100101'),bitarray('11000011'),
				bitarray('10010110')]


		# 8 random streams for all but center pixel
		# 4 random streams for copies of
		# - z2, z4, z6 and z8 for vertical/horizontal
		# - z1, z3, z7 and z9 for diagonal sobel
		rng_z_1 = 8*[0]
		for i in range(8):
			rng_z_1[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))
		#rng_z_1 = [bitarray('00010011'),
		#		   bitarray('11101101'),
		#		   bitarray('00110110'),
		#		   bitarray('00100101'),
		#		   bitarray('01001101'),
		#		   bitarray('10110010'),
		#		   bitarray('11100110'),
		#		   bitarray('00111100')]

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
				Gxy = sobelFilter(workingArea,r_had,rng_z_1,errRate)
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
		global lfsrSize
		global lfsr_seeds
		global auxStr
		r = 5*[0]
		root_lfsr = bitarray('1010010110100101')
		#5 numbers from 16x16 hadamard matrix that have SCC = 0
		#r_had =[bitarray('1001100110011001'),bitarray('1111000011110000'),
		#		bitarray('1010010110100101'),bitarray('1100001111000011'),
		#		bitarray('1001011010010110')]
		r_had =[bitarray('10011001'),bitarray('11110000'),
				bitarray('10100101'),bitarray('11000011'),
				bitarray('10010110')]

		# 8 random streams for all but center pixel
		# 4 random streams for copies of
		# - z2, z4, z6 and z8 for vertical/horizontal
		# - z1, z3, z7 and z9 for diagonal sobel
		rng_z_1 = 8*[0]
		for i in range(8):
			rng_z_1[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))
		#rng_z_1 = [bitarray('00010011'),
		#		   bitarray('11101101'),
		#		   bitarray('00110110'),
		#		   bitarray('00100101'),
		#		   bitarray('01001101'),
		#		   bitarray('10110010'),
		#		   bitarray('11100110'),
		#		   bitarray('00111100')]
		
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
				Gxy = sobelFilter(workingArea,r_had,rng_z_1,errRate)
				xy_image[i-1][j-1] = Gxy

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

def saveHex(filename=None,numpyarray=np.zeros(1)):
	'''Saves image in memory in NumPy's uint8 format
	into a text file with two hexadecimal characters
	txt format is preferrable.
	'''
	if(not filename):
		print("Invalid filename")
		return 0
	np.savetxt(filename,numpyarray,"%.2x")

def loadHex(filename=None):
	'''Reads a txt file containing a NumPy image in uint8
	format saved in pairs of hexadecimal characters.
	Returns the image as a NumPy uint8 array.
	'''
	if(not filename):
		print("Invalid hex filename")
		return None
	elif(isinstance(type(filename),str)):
		print("Invalid hex filename")
		return None
	elif(filename[-3:]!='txt'):
		print("Invalid hex filename")
		return None
	else:
		#Load images stored in hex txt files
		input_img = open(filename)

		#Put the contents into list of strings
		input_str = input_img.readlines()

		#Close file after reading
		input_img.close()

		#Prepare matrices
		decoded_mat = []

		#Take away trailing new line('\n') and convert to
		#List of byte arrays
		for i in range(len(input_str)):
			input_str[i] = input_str[i].rstrip('\n')
			decoded_mat.append(bytearray.fromhex(input_str[i]))

		#Convert bytes into unsigned 8-bit integers
		#This is one of the compatible image formats
		decoded_mat = np.array(decoded_mat,np.uint8)
		return decoded_mat

def showHex(filename=None):
	'''Reads a txt file containing a NumPy image in uint8
	format saved in pairs of hexadecimal characters.
	Shows the image on the screen and returns the
	image as a NumPy uint8 array.
	'''
	if(not filename):
		print("Invalid hex filename")
		return 0
	elif(isinstance(type(filename),str)):
		print("Invalid hex filename")
		return None
	elif(filename[-3:]!='txt'):
		print("Invalid hex filename")
		return 0
	else:
		#Load image stored in hex txt files
		input_img = open(filename)

		#Put the contents into list of strings
		input_str = input_img.readlines()

		#Close file after reading
		input_img.close()

		#Prepare matrices
		decoded_mat = []

		#Take away trailing new line('\n') and convert to
		#List of byte arrays
		for i in range(len(input_str)):
			input_str[i] = input_str[i].rstrip('\n')
			decoded_mat.append(bytearray.fromhex(input_str[i]))

		#Convert bytes into unsigned 8-bit integers
		#This is one of the compatible image formats
		decoded_mat = np.array(decoded_mat,np.uint8)

		#Plot the image
		plt.imshow(decoded_mat,cmap='gray')
		plt.show()
		return decoded_mat
