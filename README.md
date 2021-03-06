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

## Quick-start
All commands consider a Linux environment

### Deterministic implementation
```bash
cd ./sw-sim
python interactiveDetSobel.py
src, edges = detectAndShow('320px-1000_years_Old_Thanjavur_Brihadeeshwara_Temple_View_at_Sunrise.jpg')
```
This will calculate an edge image in a few seconds and plot the result in a new window. It also returns the source image as a numpy array (`src`) and the 8-bit `edges` array.
Expected image:
![Temple View edge image processed by deterministic algorithm](./320px_det_edges.png?raw=true "Temple View edge image processed by deterministic algorithm")

### Stochastic implementation
```bash
cd ./sw-sim
python interactiveStochSobel.py
src, edges = detectAndShow('320px-1000_years_Old_Thanjavur_Brihadeeshwara_Temple_View_at_Sunrise.jpg')
```
This version takes more time than the deterministic version since Python types and code have not been optimized to do the bit-wise operations that Stochastic Computer takes advantage of.
Expected image:
![Temple View edge image processed by stochastic algorithm](./320px_stoch_edges.png?raw=true "Temple View edge image processed by stochastic algorithm")

#### Parallel processing with Ray
If available, you may opt to use
```bash
python interactiveRayStochSobel.py
```
instead. This version defaults to starting 8 threads to process images, which is about 8 times faster than the single-threaded version on modern CPUs (which already come with 8 threads).

## Remarks:
* Deterministic implementation:
	* Works as OpenCV's Sobel example without image blurring
* Image database for testing from ["CURE-OR: Challenging Unreal and Real Environment for Object Recognition", IEEE Dataport, 2019. [Online]. Available: http://dx.doi.org/10.21227/h4fr-h268. Accessed: Nov. 11, 2019.](https://ieee-dataport.org/open-access/cure-or-challenging-unreal-and-real-environment-object-recognition)
* Sample image file [320px-1000_years_Old_Thanjavur_Brihadeeshwara_Temple_View_at_Sunrise.jpg](https://commons.wikimedia.org/wiki/File:1000_years_Old_Thanjavur_Brihadeeshwara_Temple_View_at_Sunrise.jpg) from Wikimedia Commons. Author: [KARTY JazZ](https://commons.wikimedia.org/wiki/User:KARTY_JazZ).

## Dependencies
Python dependencies:
* Python >= 3.9
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
Hardware simulation dependencies:
* Icarus Verilog 10.1
* `make` (for ease of executing multiple commands)

This project currently uses stochastic circuits derived from ones synthesized with [scsynth/STRAUSS](https://github.com/arminalaghi/scsynth)

Installation recommendation:
* Newer versions of Python 3 (like 3.8.x) come with pip preinstalled. [PyPi/pip](https://pypi.org/) is a simple package manager for Python (normally aliased as pip3)
* For this project:
```bash
pip3 install --user wheel numpy scipy matplotlib opencv-python bitarray 'ray[default]'
```

