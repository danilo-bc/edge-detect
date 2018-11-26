# Author: Danilo Cavalcanti
# Importing System Modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Import personal implementation library
from sobelCustom import *
print("--------------------------------------------------------")
print("--------------------------------------------------------")
print("Sucessfully loaded personal Sobel Filter implementation")
print("How to demo: ")
print("Use function 'detectAndShow(path)'")
print("Example: edges = detectAndShow('/home/images/photo.jpg')")
print("Result: photo printed on screen with edges detected and")
print("image saved into 'edges' variable as numpy matrix")
print("--------------------------------------------------------")
print("--------------------------------------------------------")
import code
code.interact(local=locals())
