import unittest
import sys
sys.path.append('../') # Append source-file folder
import stochLFSR as lfsr
from bitarray import bitarray

class Test_stochLFSR_behaviour(unittest.TestCase):
    def test_4_bit_lfsr(self):
        oracle = open("lfsr4bOracle.data",'r')
        seed=('1000')
        lfsr_reg = bitarray(seed)

        for i in range(16):
            expected = bitarray(oracle.readline()[:-1])
            lfsr_reg = lfsr.shift(lfsr_reg)
            self.assertEqual(expected, lfsr_reg)
        oracle.close()

    def test_8_bit_lfsr(self):
        seed=('10000000')
        oracle = open("lfsr8bOracle.data",'r')
        lfsr_reg = bitarray(seed)

        for i in range(16):
            expected = bitarray(oracle.readline()[:-1])
            lfsr_reg = lfsr.shift(lfsr_reg)
            print(i, lfsr_reg)
            self.assertEqual(expected, lfsr_reg)
        oracle.close()

if __name__=='__main__':
    unittest.main()