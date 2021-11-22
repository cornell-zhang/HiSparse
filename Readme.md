# <u>Hi</u>gh-Performance <u>Sparse</u> Linear Algebra on HBM-Equipped FPGAs Using HLS: A Case Study on SpMV (HiSparse)

This work is accepted by FPGA 2022.

## To reproduce the results
### 1. Colne this repo and download the datasets
```
git clone https://github.com/yxd97/HiSparse.git
cd datasets
source download.sh
```
You will find two directories: ```graph``` and ```pruned_nn``` containing the datasets used in our evaluation.

### 2. Install cnpy to load the datasets
Cnpy is a C++ library that enables reading ```.npy``` files in C++. It is open-sourced available here: https://github.com/rogersce/cnpy.
Please follow the instructions in the cnpy repo to install it.

After installing cnpy, remember to setup the following variables to load this library:
```
export CNPY_INCLUDE=<the directory contains cnpy header (cnpy.h)>
export CNPY_LIB=<the directory contains cnpy library (libcnpy.so)>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CNPY_LIB
```

### 3. Set up Xilinx Vitis 2020.2
This step is defferent depending on the installation setup on your machine.
However, to check whether you have correctly set it up, you can do
```
printenv VITIS
```
the path to the Vitis installation should appear if it's correctly set up.

### 4. Build and run the design
```
cd sw
make benchmark IMPL=<fixed/float_pob/float_stall>
```
The ```IMPL``` option is used to switch between
the fixed-point design,
the floating-point deisgn using partial output buffers,
and the floating-point design using stall + row interleaving.