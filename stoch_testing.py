# Author: Danilo Cavalcanti
# Importing System Modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Import personal implementation library
from sobelCustomStoch import *
"""
print('''------------------------------------------------------------------
------------------------------------------------------------------
Sucessfully loaded personal Sobel Filter implementation
How to demo:
Use functions 'detectAndShow(image)' or 'detectAndWritePGM(image)'
Example: src,edges = detectAndShow('./images/aerial/school.pgm')
Result: photo printed on screen with edges detected,
base image returned as 'src' and
edge image returned into 'edges' variable as numpy matrix
------------------------------------------------------------------
------------------------------------------------------------------''')
"""
lfsr_4_bit = lfsr(4,'0000')
for i in range(16):
	print(lfsr_4_bit.shift())

import code
code.interact(local=locals())
