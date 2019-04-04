# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray

lfsr_len = 4
# auxiliary variable to mimic Verilog behaviour
aux = lfsr_len-1

lfsr_val = bitarray(lfsr_len*'0')
seed = lfsr_len*'0'

def lfsr():
    global lfsr_val

    xor_op = lfsr_val[aux-3] ^ lfsr_val[aux-2]
    zero_detector = ~(lfsr_val[1:])
    shift_in = xor_op ^ zero_detector.all() #lfsr_val[-1]
    lfsr_val.append(shift_in)
    del lfsr_val[0]

    return lfsr_val

def lfsr_set_seed(seed_in):
    '''
    Sets LFSR's seed given in a string
    e.g.: '1101'
    '''
    global seed
    seed = seed_in

def lfsr_reset():
    global lfsr_val
    lfsr_val = bitarray(lfsr_len*'0')

def lfsr_restart():
    global lfsr_val
    global seed
    lfsr_val = bitarray(seed)