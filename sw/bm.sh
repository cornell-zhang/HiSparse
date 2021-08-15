#!/bin/bash

DATASET_PATH=/work/shared/common/project_build/graphblas/data/sparse_matrix_graph

# DATASETS=(gplus_108K_13M_csr_float32.npz
#           ogbl_ppa_576K_42M_csr_float32.npz
#           hollywood_1M_113M_csr_float32.npz
#           pokec_1633K_31M_csr_float32.npz
#           ogbn_products_2M_124M_csr_float32.npz
#           orkut_3M_213M_csr_float32.npz)

DATASETS=(orkut_3M_213M_csr_float32.npz)

bitstream=/work/shared/common/project_build/graphblas/graphlily-synthesize/spmv-syn/obl3/build_dir.hw/spmv.xclbin

for ((i = 0; i < ${#DATASETS[@]}; i++)) do
    ./benchmark $bitstream $DATASET_PATH/${DATASETS[i]}
done
