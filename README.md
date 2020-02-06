# Edge Detection

Sobel edge detection algorithms implemented in Python for grayscale images.

This project intends to serve as a model for future Stochastic Computing applications.

## Progress
- [x] Deterministic
  - [x] Software
  - [x] Hardware
- [x] Stochastic
  - [x] Software
    - [x] LFSR
    - [x] Module
    - [x] Wrapper
  - [x] Hardware

Remarks:
* Deterministic implementation:
	* Works as OpenCV's Sobel example without image blurring
* Stochastic implementation:
  * Hardware implementation needs to be reviewed
* Image database for testing from ["CURE-OR: Challenging Unreal and Real Environment for Object Recognition", IEEE Dataport, 2019. [Online]. Available: http://dx.doi.org/10.21227/h4fr-h268. Accessed: Nov. 11, 2019.](https://ieee-dataport.org/open-access/cure-or-challenging-unreal-and-real-environment-object-recognition)

Python dependencies:
* Python >= 3.6.7
* Libraries:
  * opencv-python
  * matplotlib
  * numpy
  * scipy
  * bitarray
  * ray (recommended)
    * for parallel processing in simulation
    * currently, only supported on Linux and MacOS
  * wheel (recommended for bitarray support)
  * setproctitle (optional)
  * psutil (optional)
  * aiohttp (optional)
  * grpcio (optional)

Hardware simulation dependencies:
* Icarus Verilog 10.1
* make (for ease of executing multiple commands)

This project currently uses stochastic circuits derived from ones synthesized with [scsynth/STRAUSS](https://github.com/arminalaghi/scsynth)

Installation recommendation:
* Newer versions of Python 3 (like 3.8.x) come with pip preinstalled. [PyPi/pip](https://pypi.org/) is a simple package manager for Python (normally aliased as pip3)
* For this project:
```bash
pip3 install --user numpy scipy matplotlib wheel opencv-python ray bitarray setproctitle psutil aiohttp grpcio
```

