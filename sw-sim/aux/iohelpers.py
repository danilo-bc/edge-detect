# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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


def hexToPGM(filename=None,outputfile=None):
	'''Reads a txt file containing a NumPy image in uint8
	format saved in pairs of hexadecimal characters.
	Shows the image on the screen and returns the
	image as a NumPy uint8 array.
	'''
	if(not filename or not outputfile):
		print("Invalid filename")
		return 0
	elif(isinstance(type(filename),str) and isinstance(type(outputfile),str)):
		print("Invalid filename")
		return None
	elif(filename[-3:]!='txt'):
		print("Invalid hex filename")
		return 0
	elif(outputfile[-3:]!='pgm'):
		print("Invalid PGM filename")
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

		flag1 = cv.imwrite(outputfile,decoded_mat)
		if(flag1):
			print('''File''',outputfile,''' successfully created''')
		else:
			print('''Something went wrong during the saving process''')

	return decoded_mat