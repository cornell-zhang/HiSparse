# <u>Hi</u>gh-Performance <u>Sparse</u> Linear Algebra on HBM-Equipped FPGAs Using HLS: A Case Study on SpMV (HiSparse)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5819246.svg)](https://doi.org/10.5281/zenodo.5819246)

HiSparse is a high-performance accelerator for sparse-matrix vcetor multiplication (SpMV).
Implemented on a multi-die HBM-equipped FPGA device, HiSparse achieves 237MHz and delivers promising
speedup with increased bandwidth efficiency when compared to prior arts on CPUs, GPUs, and FPGAs.

For more information, please refer to [our FPGA 2022 paper](https://github.com/cornell-zhang/HiSparse/blob/master/fpgafp193a-du.pdf).
```
@article{du2022hisparse,
  title={{High-Performance Sparse Linear Algebra on HBM-Equipped FPGAs Using HLS: A Case Study on SpMV}},
  author={Du, Yixiao and Hu, Yuwei and Zhou, Zhongchun and Zhang, Zhiru},
  journal={{Int'l Symp. on Field-Programmable Gate Arrays (FPGA)}},
  year={2022}
}
```

## Prerequisites
* Platform: Xilinx Alveo U280
* Toolchain: Xilinx Vitis 2020.2

## To reproduce the results
### 1. Colne this repo and download the datasets
```
git clone https://github.com/cornell-zhang/HiSparse.git
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

### 4. Run the pre-complied bitstream
This repo has a pre-complied fixed-point bitstream: ```dempo_spmv.xclbin```.
You can directly run it using
```
cd sw
make demo
```
The benchmark results will be printed out as the program is running, in the format as:
```
{Preprocessing: 0.64566 s | SpMV: 0.77102 ms | 49.4087 GBPS | 12.9698 GOPS }
```
The numbers are: pre-processing time, SpMV run time, SpMV data throughput, SpMV operation throughput.

Note: data throughput = operation throughput / 2 * 8.

### 5. Build and run the design
```
cd sw
make benchmark IMPL=<fixed/float_pob/float_stall>
```
The ```IMPL``` option is used to switch between
the fixed-point design,
the floating-point deisgn using partial output buffers,
and the floating-point design using stall + row interleaving.
