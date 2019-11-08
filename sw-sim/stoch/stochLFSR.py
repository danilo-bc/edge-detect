# Author: Danilo Cavalcanti
# Importing System Modules
from bitarray import bitarray

def shift(lfsr_reg):
	'''
	the location of XORs obtained from http://www.newwaveinstruments.com
	/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.h
	tm. If no taps are supplied, taps will be autamatically selected for
	lfsrLen between 3 and 20. 24. or 32
	'''
	# Aliases for reg and len(reg) to shorten code
	a = len(lfsr_reg)
	v = lfsr_reg
	xorOp = False
	if(a==3):
		xorOp = v[a-3]^v[a-2]
	elif(a==4):
		xorOp = v[a-4]^v[a-3]
	elif(a==5):
		xorOp = v[a-5]^v[a-3]
	elif(a==6):
		xorOp = v[a-6]^v[a-5]
	elif(a==7):
		xorOp = v[a-7]^v[a-6]
	elif(a==8):
		xorOp = v[a-8]^v[a-7]^v[a-6]^v[a-1]
	elif(a==9):
		xorOp = v[a-9]^v[a-5]
	elif(a==10):
		xorOp = v[a-10]^v[a-7]
	elif(a==11):
		xorOp = v[a-11]^v[a-9]
	elif(a==12):
		xorOp = v[a-12]^v[a-11]^v[a-10]^v[a-4]
	elif(a==13):
		xorOp = v[a-13]^v[a-12]^v[a-11]^v[a-8]
	elif(a==14):
		xorOp = v[a-14]^v[a-13]^v[a-12]^v[a-2]
	elif(a==15):
		xorOp = v[a-15]^v[a-14]
	elif(a==16):
		xorOp = v[a-16]^v[a-15]^v[a-13]^v[a-4]
	elif(a==17):
		xorOp = v[a-17]^v[a-14]
	elif(a==18):
		xorOp = v[a-18]^v[a-11]
	elif(a==19):
		xorOp = v[a-19]^v[a-18]^v[a-17]^v[a-14]
	elif(a==20):
		xorOp = v[a-20]^v[a-17]
	elif(a==24):
		xorOp = v[a-24]^v[a-23]^v[a-22]^v[a-17]
	elif(a==32):
		xorOp = v[a-32]^v[a-31]^v[a-30]^v[a-10]
	else:
		xorOp = v[a-3]^v[a-2]
	zeroDetector = ~(v[1:])
	shiftIn = xorOp ^ zeroDetector.all()
	v.append(shiftIn)
	del v[0]

	return v

def next(lfsr_reg):
	aux = shift(lfsr_reg)
	return bool(int(aux[0]))
"""
class lfsr:
	def __init__(self,lfsrLen = None,seed = None):
		if(lfsrLen == None):
			lfsrLen = 8
		else:
			lfsrLen=lfsrLen
		if(seed == None):
			seed = lfsrLen*'0'
		else:
			seed = seed
		restart()

	def shift(self):
		defineXorOp()
		zeroDetector = ~(lfsrVal[1:])
		shiftIn = xorOp ^ zeroDetector.all()
		lfsrVal.append(shiftIn)
		del lfsrVal[0]

		return lfsrVal.to01()

	def next(self):
		aux = shift()
		return bool(int(aux[0]))

	def setSeed(self,seed_in):
		'''
		Sets LFSR's seed given in a string
		e.g.: '1101'
		'''
		seed = seed_in

	def reset(self):
		lfsrVal = bitarray(lfsrLen*'0')

	def restart(self):
		lfsrVal = bitarray(seed)

	def defineXorOp(self):
		'''
		the location of XORs obtained from http://www.newwaveinstruments.com
        /resources/articles/m_sequence_linear_feedback_shift_register_lfsr.h
        tm. If no taps are supplied, taps will be autamatically selected for
        lfsrLen between 3 and 20. 24. or 32
		'''
		# Aliases for lfsrLen and lfsrVal to shorten code
		a = lfsrLen
		v = lfsrVal

		if(a==3):
			xorOp = v[a-3]^v[a-2]
		elif(a==4):
			xorOp = v[a-4]^v[a-3]
		elif(a==5):
			xorOp = v[a-5]^v[a-3]
		elif(a==6):
			xorOp = v[a-6]^v[a-5]
		elif(a==7):
			xorOp = v[a-7]^v[a-6]
		elif(a==8):
			xorOp = v[a-8]^v[a-7]^v[a-6]^v[a-1]
		elif(a==9):
			xorOp = v[a-9]^v[a-5]
		elif(a==10):
			xorOp = v[a-10]^v[a-7]
		elif(a==11):
			xorOp = v[a-11]^v[a-9]
		elif(a==12):
			xorOp = v[a-12]^v[a-11]^v[a-10]^v[a-4]
		elif(a==13):
			xorOp = v[a-13]^v[a-12]^v[a-11]^v[a-8]
		elif(a==14):
			xorOp = v[a-14]^v[a-13]^v[a-12]^v[a-2]
		elif(a==15):
			xorOp = v[a-15]^v[a-14]
		elif(a==16):
			xorOp = v[a-16]^v[a-15]^v[a-13]^v[a-4]
		elif(a==17):
			xorOp = v[a-17]^v[a-14]
		elif(a==18):
			xorOp = v[a-18]^v[a-11]
		elif(a==19):
			xorOp = v[a-19]^v[a-18]^v[a-17]^v[a-14]
		elif(a==20):
			xorOp = v[a-20]^v[a-17]
		elif(a==24):
			xorOp = v[a-24]^v[a-23]^v[a-22]^v[a-17]
		elif(a==32):
			xorOp = v[a-32]^v[a-31]^v[a-30]^v[a-10]
		else:
			xorOp = v[a-3]^v[a-2]
"""
"""
def test(self):
	# Test 4-bit LFSR
	oracle = open("lfsr4b8bOracle.data",'r')
	seed=('1000')
	lfsrLen=4
	restart()
	fail4 = False
	print("Beginning 4-bit LFSR test")
	for i in range(16):
		exp = bitarray(oracle.readline()[:-1])
		eval = shift()
		try:
			assert (eval == exp.to01())
		except AssertionError:
			print("Error line: ",i+1,":\tOutput: "+eval+"\tExpected: "+exp.to01())
			fail4 = True

	# Test 8-bit LFSR
	seed=('10000000')
	lfsrLen=8
	restart()
	fail8 = False
	print("Beginning 8-bit LFSR test")
	for i in range(16):
		exp = bitarray(oracle.readline()[:-1])
		eval = shift()
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
		"""

