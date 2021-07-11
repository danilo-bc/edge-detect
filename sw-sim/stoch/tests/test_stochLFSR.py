import unittest
import sys
sys.path.append('../') # Append source-file folder
import stochLFSR as lfsr
from bitarray import bitarray

class Test_stochLFSR_behaviour(unittest.TestCase):
    def test_4_bit_lfsr(self):
        # Reference sequence from hw-sim/LFSR-tb
        expected_sequence = ['0000','0001','0010','0100','1001','0011','0110','1101','1010','0101','1011','0111','1111','1110','1100','1000']
        seed=('1000')
        lfsr_reg = bitarray(seed)

        for i in range(16):
            expected = bitarray(expected_sequence[i])
            lfsr_reg = lfsr.shift(lfsr_reg)
            self.assertEqual(expected, lfsr_reg)

    def test_8_bit_lfsr(self):
        seed=('10000000')
        expected_sequence = ['00000000','00000001','00000011','00000111','00001111','00011111','00111111','01111110','11111100','11111001','11110010','11100101','11001010','10010100','00101001','01010010']
        lfsr_reg = bitarray(seed)

        for i in range(16):
            expected = bitarray(expected_sequence[i])
            lfsr_reg = lfsr.shift(lfsr_reg)
            print(i, lfsr_reg)
            self.assertEqual(expected, lfsr_reg)

if __name__=='__main__':
    unittest.main()