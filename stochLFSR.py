# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray

class lfsr:
	def __init__(self,lfsrLen = None,seed = None):
		if(lfsrLen == None):
			self.lfsrLen = 8
		else:
			self.lfsrLen=lfsrLen
		if(seed == None):
			self.seed = self.lfsrLen*'0'
		else:
			self.seed = seed
		self.restart()

	def shift(self):
		self.defineXorOp()
		zeroDetector = ~(self.lfsrVal[1:])
		shiftIn = self.xorOp ^ zeroDetector.all()
		self.lfsrVal.append(shiftIn)
		del self.lfsrVal[0]

		return self.lfsrVal.to01()

	def next(self):
		aux = self.shift()
		return bool(int(aux[0]))

	def setSeed(self,seed_in):
		'''
		Sets LFSR's seed given in a string
		e.g.: '1101'
		'''
		self.seed = seed_in

	def reset(self):
		self.lfsrVal = bitarray(lfsrLen*'0')

	def restart(self):
		self.lfsrVal = bitarray(self.seed)

	def defineXorOp(self):
		'''
		the location of XORs obtained from http://www.newwaveinstruments.com
        /resources/articles/m_sequence_linear_feedback_shift_register_lfsr.h
        tm. If no taps are supplied, taps will be autamatically selected for
        lfsrLen between 3 and 20. 24. or 32
		'''
		# Aliases for lfsrLen and lfsrVal to shorten code
		a = self.lfsrLen
		v = self.lfsrVal

		if(a==3):
			self.xorOp = v[a-3]^v[a-2]
		elif(a==4):
			self.xorOp = v[a-4]^v[a-3]
		elif(a==5):
			self.xorOp = v[a-5]^v[a-3]
		elif(a==6):
			self.xorOp = v[a-6]^v[a-5]
		elif(a==7):
			self.xorOp = v[a-7]^v[a-6]
		elif(a==8):
			self.xorOp = v[a-8]^v[a-7]^v[a-6]^v[a-1]
		elif(a==9):
			self.xorOp = v[a-9]^v[a-5]
		elif(a==10):
			self.xorOp = v[a-10]^v[a-7]
		elif(a==11):
			self.xorOp = v[a-11]^v[a-9]
		elif(a==12):
			self.xorOp = v[a-12]^v[a-11]^v[a-10]^v[a-4]
		elif(a==13):
			self.xorOp = v[a-13]^v[a-12]^v[a-11]^v[a-8]
		elif(a==14):
			self.xorOp = v[a-14]^v[a-13]^v[a-12]^v[a-2]
		elif(a==15):
			self.xorOp = v[a-15]^v[a-14]
		elif(a==16):
			self.xorOp = v[a-16]^v[a-15]^v[a-13]^v[a-4]
		elif(a==17):
			self.xorOp = v[a-17]^v[a-14]
		elif(a==18):
			self.xorOp = v[a-18]^v[a-11]
		elif(a==19):
			self.xorOp = v[a-19]^v[a-18]^v[a-17]^v[a-14]
		elif(a==20):
			self.xorOp = v[a-20]^v[a-17]
		elif(a==24):
			self.xorOp = v[a-24]^v[a-23]^v[a-22]^v[a-17]
		elif(a==32):
			self.xorOp = v[a-32]^v[a-31]^v[a-30]^v[a-10]
		else:
			self.xorOp = v[a-3]^v[a-2]

	def test(self):
		# Test 4-bit LFSR
		oracle = open("lfsr4b8bOracle.data",'r')
		self.seed=('1000')
		self.lfsrLen=4
		self.restart()
		fail4 = False
		print("Beginning 4-bit LFSR test")
		for i in range(16):
			exp = bitarray(oracle.readline()[:-1])
			eval = self.shift()
			try:
				assert (eval == exp.to01())
			except AssertionError:
				print("Error line: ",i+1,":\tOutput: "+eval+"\tExpected: "+exp.to01())
				fail4 = True

		# Test 8-bit LFSR
		self.seed=('10000000')
		self.lfsrLen=8
		self.restart()
		fail8 = False
		print("Beginning 8-bit LFSR test")
		for i in range(16):
			exp = bitarray(oracle.readline()[:-1])
			eval = self.shift()
			try:
				assert (eval == exp.to01())
			except AssertionError:
				print("Error line: ",i+17,":\tOutput: "+eval+"\tExpected: "+exp.to01())
				fail8 = True

		# Print if succesfull results
		if(fail4 ==False):
			print("4-bit test succesfull")
		if(fail8 ==False):
			print("8-bit test succesfull")
		oracle.close()

