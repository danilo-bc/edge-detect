# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray
from scipy.stats import bernoulli

from aux.auxi import *

# Making constants the Python way
# "Making them in a separate file and importing
# them on a main.py" thing
sKernelX=np.array([[-1,0,1],
						[-2,0,2],
						[-1,0,1]],np.float64)

sKernelY=np.array([[-1,-2,-1],
						[0,0,0],
						[1,2,1]],np.float64)

def sobelFilter(img=-1,errRate=0.0):
	'''Alternate function that calculates Gx and Gy of a 3x3
	img in numpy matrix form to compare with Stochastic version
	Arguments:
	- img: 3x3 region to process Gx and Gy
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning default (0)")
		return 0
	elif(img.shape!=(3,3)):
		print("Invalid 'img' shape (not 3x3), returning default (0)")
		return 0
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning default (0)")
		return 0
	elif(errRate<0.0 or errRate>1.0):
		print("Invalid error rate, must be between 0.0 and 1.0")
		return 0
	else:
		Gx = np.float64(0)
		Gy = np.float64(0)

		# Do the convolution in one of NumPy's way
		Gx = np.sum(sKernelX*img)
		Gy = np.sum(sKernelY*img)

		ans = np.uint8((0.25*np.abs(Gx)+0.25*np.abs(Gy))/2.0)
		if(errRate!=0):
			ansBin = bitarray('{0:08b}'.format(ans))
			for i in range(len(ansBin)):
				errBit = bernoulli.rvs(errRate,size=1)
				ansBin[i] = ansBin[i]^errBit
			ans = int(ansBin.to01(),2)

		return np.uint8(ans)

def createEdgeImage(img=-1,errRate=0.0):
	''' Applies Sobel filter on a NxM image "img" loaded via OpenCV (cv2 package) and
	returns three (N-2)x(M-2) images with sobelX, sobelY and both filters applied.
	2 rows and 2 columns removed to simplify boundary conditions.
	Arguments:
	- img: Region in Grayscale color scheme and np.uint8 format
	'''
	if(type(img) != np.ndarray):
		print("Invalid 'img' parameter, returning empty matrix")
		return np.array([0],np.float64)
	elif(img.dtype != np.float64):
		print("Invalid 'img' dtype (not float64), returning empty matrix")
		return np.array([0],np.float64)
	elif(errRate<0.0 or errRate>1.0):
		print("Invalid error rate, must be between 0.0 and 1.0")
	else:
		# Create images ignoring last row and column for simplicity in
		# convolution operation

		xy_image = np.zeros([img.shape[0]-2,img.shape[1]-2])
		for i in range(1,img.shape[0]-1):
			for j in range(1,img.shape[1]-1):
				# Reset Gx & Gy for each pixel
				Gx, Gy = 0, 0
				# Get 3x3 submatrixes with np.ix_
				# [i-1,i,i+1] = range(i-1,i+2)
				# kept explicit for clarity
				ixgrid = np.ix_([i-1,i,i+1],[j-1,j,j+1])
				workingArea = img[ixgrid]
				# Call the convolution function
				xy_image[i-1][j-1] = sobelFilter(workingArea,errRate)

		return xy_image

def detectAndShow(imgpath=0,errRate=0.0,show=True):
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

	# Plot side by side for comparison
	#plt.subplot(1,2,1), plt.imshow(img,cmap='gray')
	#plt.title('Original'), plt.xticks([]), plt.yticks([])
	#plt.subplot(1,2,2), plt.imshow(xy_img,cmap='gray')
	#plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

	# Plot only results
	if(show):
		plt.imshow(xy_img,cmap='gray')
		plt.show()

	return np.uint8(img),np.uint8(xy_img)

def detectAndWritePGM(imgpath=0):
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

	# Write back into PGM image files
	flag1 = cv.imwrite("src.pgm",img)
	flag2 = cv.imwrite("edges.pgm",xy_img)
	if(flag1 and flag2):
		print('''Files "src.pgm" and "edges.pgm" successfully created''')
	else:
		print('''Something went wrong during the saving process''')

	return img,xy_img

def opencvSobelFilter(img=-1):
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
		Gx = np.float64(0.0)
		Gy = np.float64(0.0)

		# Do the convolution in one of NumPy's way
		Gx = np.sum(sKernelX*img)
		Gy = np.sum(sKernelY*img)

		return Gx,Gy

def opencvCreateEdgeImage(img=-1):
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
		x_image = np.zeros([img.shape[0]-2,img.shape[1]-2])
		y_image = np.zeros([img.shape[0]-2,img.shape[1]-2])

		xy_image = np.zeros([img.shape[0]-2,img.shape[1]-2])
		for i in range(1,img.shape[0]-1):
			for j in range(1,img.shape[1]-1):
				# Reset Gx & Gy for each pixel
				Gx, Gy = 0, 0
				# Get 3x3 submatrixes with np.ix_
				# [i-1,i,i+1] = range(i-1,i+2)
				# kept explicit for clarity
				ixgrid = np.ix_([i-1,i,i+1],[j-1,j,j+1])
				workingArea = img[ixgrid]
				# Call the convolution function
				Gx, Gy = sobelFilter(workingArea)
				x_image[i-1][j-1] = Gx
				y_image[i-1][j-1] = Gy

		# Take absolute value and
		x_image = np.abs(x_image)
		y_image = np.abs(y_image)
		# Saturate x and y_image to fit 8-bit
		x_image[x_image>255] = 255
		y_image[y_image>255] = 255

		# Sum halves to keep results in [0-255]
		xy_image = 0.5*x_image+0.5*y_image
		return xy_image

def opencvDetectAndShow(imgpath=0):
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
	xy_img = opencvCreateEdgeImage(np.array(img,np.float64))

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

