# Edge Detection

Sobel edge detection algorithms implemented in Python for grayscale images.

This project intends to serve as a model for future Stochastic Computing applications.

## Progress
- [x] Deterministic
  - [x] Software
  - [x] Hardware
- [ ] Stochastic
  - [ ] Software
    - [x] LFSR
    - [ ] Module
    - [ ] Wrapper
  - [ ] Hardware

Remarks:
* Deterministic implementation:
	* Currently requires minor revision on image normalization
* Stochastic implementation
  * Software LFSR works as hardware implementation
  * Hardware implementation needs to be reviewed
* Image database for testing from [K. Bowyer, C. Kranenburg, and S. Dougherty, "Edge Detector Evaluation
   Using Empirical ROC Curves", Computer Vision and Pattern Recognition '99,
   Fort Collins, Colorado. Vol 1, pp 354-359. June 1999.](http://figment.csee.usf.edu/edge/roc/)

Python dependencies:
* Python 3.6.7
* opencv, matplotlib, numpy and bitarray

Hardware simulation dependencies:
* Icarus Verilog 10.1
* make (for ease of executing multiple bash commands)

This project currently uses stochastic circuits synthesized from [scsynth/STRAUSS](https://github.com/arminalaghi/scsynth)

Installation recommendation:
* Newer versions of Python 3 (like 3.6.x) come with pip preinstalled. [PyPi/pip](https://pypi.org/) is a simple package manager for Python (normally aliased as pip3)
* For this project:
```bash
pip3 install numpy
pip3 install opencv-python
pip3 install matplotlib
pip3 install bitarray
```

