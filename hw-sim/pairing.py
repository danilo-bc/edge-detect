import math
import numpy as np

def pair(k1, k2, safe=True):
    """
    Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    """
    z = int(0.5 * (k1 + k2) * (k1 + k2 + 1) + k2)
    if safe and (k1, k2) != depair(z):
        raise ValueError("{} and {} cannot be paired".format(k1, k2))
    return z
def depair(z):
    """
    Inverse of Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
    """
    w = math.floor((math.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    # assert z != pair(x, y, safe=False):
    return x, y

def pairSC(N,i,j):
    res = int(0.5 * (i+j) * (i+j+1) + j)
    return res

N = 256
degrees = [1,1,1,1,1,1,1,1]

orig = []
paired = []
mod = []
for i in range(1,len(degrees)+1,1):
    for j in range(1):
        orig.append(round((0.5*(i+j)*(i+j+1)+j)/(1*2+1)))
        paired.append(pair(i,j))
        mod.append(pairSC(N,i,j))
#normalize mod
rmin = np.min(mod)
rmax = np.max(mod)
tmin = 0
tmax = N-2
mod = (mod - rmin)*(tmax-tmin)/(rmax-rmin)+tmin
mod = np.intc(np.round(mod))

print(orig)
print(paired)
print(mod)
print(len(set(np.intc(np.round(mod)))))
# Existing code goes above this
import code
code.interact(local=locals())
# Exit interaction using CTRL+D (EOF)
