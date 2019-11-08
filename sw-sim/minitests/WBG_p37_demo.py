# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray

len = 4
lfsrVal = bitarray('1001')
num = bitarray('1011')

def shift():
	global lfsrVal
	zeroDetector = ~(lfsrVal[1:])
	shiftIn =  lfsrVal[len-4]^ lfsrVal[len-3] ^ zeroDetector.all()
	lfsrVal.append(shiftIn)
	del lfsrVal[0]

	return lfsrVal.to01()

def WBG():
	shift()
	W = 4*[0]
	W[3] = lfsrVal[len-4]
	W[2] = (~lfsrVal[len-4])&  lfsrVal[len-3]
	W[1] = (~lfsrVal[len-4])&(~lfsrVal[len-3])&  lfsrVal[len-2]
	W[0] = (~lfsrVal[len-4])&(~lfsrVal[len-3])&(~lfsrVal[len-2])& lfsrVal[len-1]

	return (W[3]&num[len-4])^(W[2]&num[len-3])^(W[1]&num[len-2])^(W[0]&num[len-1])

# number precisely turned into stochastic notation
stoch_num = ''
for i in range(16):
	stoch_num += str(WBG())
stoch_num = bitarray(stoch_num)
print("Original number: ",num, "\tValue: ",int(num.to01(),2))
print("Stochastic form: ",stoch_num, "\tValue (number of 1's): ",stoch_num.count())
