# Author: Danilo Cavalcanti
# Importing System Modules
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from bitarray import bitarray

lfsr_len = 4
lfsr_val = bitarray(lfsr_len*'0')
seed = lfsr_len*'0'

def lfsr():
    global lfsr_val

    xor_op = lfsr_val[3] ^ lfsr_val[2]
    zero_detector = 0#~(|shift[0:3])
    shift_in = lfsr_val[-1] #xor_op ^ zero_detector
    lfsr_val.insert(0,shift_in)
    del lfsr_val[-1]

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