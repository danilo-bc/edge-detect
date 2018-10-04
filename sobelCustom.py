import numpy as np
import cv2 as cv
import math as m

# Making constants the Python way
# "Making them in a separate file and importing
# them on a main.py thing"
sKernelX=np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]],np.float64)
      
sKernelY=np.array([[-1,-2,-1],
                      [0,0,0],
                      [1,2,1]],np.float64)

def sobelCustom(img=-1):
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
        
        for i in range(3):
            for j in range(3):
                Gx += sKernelX[i][j]*img[i][j]
                Gy += sKernelY[i][j]*img[i][j]
        
        return Gx,Gy


