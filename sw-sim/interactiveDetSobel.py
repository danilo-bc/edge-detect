# Author: Danilo Cavalcanti
# Importing System Modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Import personal implementation library
from det.detSobel import *
print('''------------------------------------------------------------------
------------------------------------------------------------------
Sucessfully loaded personal Sobel Filter implementation
How to demo:
Use functions 'detectAndShow(image)' or 'detectAndWritePGM(image)'
Example: src, edges = detectAndShow('800px-1000_years_Old_Thanjavur_Brihadeeshwara_Temple_View_at_Sunrise.jpg')
Result: photo printed on screen with edges detected,
base image returned as 'src' and
edge image returned into 'edges' variable as numpy matrix
------------------------------------------------------------------
------------------------------------------------------------------''')
import code
code.interact(local=locals())
