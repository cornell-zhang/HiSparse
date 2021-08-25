#!/bin/bash

GRAPH_DATASET_PATH=/work/shared/common/project_build/graphblas/data/sparse_matrix_graph
GRAPH_DATASETS=(gplus_108K_13M_csr_float32.npz
                ogbl_ppa_576K_42M_csr_float32.npz
                hollywood_1M_113M_csr_float32.npz
                pokec_1633K_31M_csr_float32.npz
                ogbn_products_2M_124M_csr_float32.npz
                mouse_gene_45K_29M_csr_float32.npz)

NN_DATASET_PATH=/work/shared/common/project_build/graphblas/data/pruned_neural_network
NN_DATASETS=(transformer_50_33288_512_csr_float32.npz
             transformer_60_33288_512_csr_float32.npz
             transformer_70_33288_512_csr_float32.npz
             transformer_80_33288_512_csr_float32.npz
             transformer_90_33288_512_csr_float32.npz
             transformer_95_33288_512_csr_float32.npz
             transformer_50_512_33288_csr_float32.npz
             transformer_60_512_33288_csr_float32.npz
             transformer_70_512_33288_csr_float32.npz
             transformer_80_512_33288_csr_float32.npz
             transformer_90_512_33288_csr_float32.npz
             transformer_95_512_33288_csr_float32.npz)

# fixed-point bitstream
# bitstream=/work/shared/common/project_build/graphblas/graphlily-synthesize/spmv-syn/obl3-fwd/build_dir.hw/spmv.xclbin
# floating-point bitstream (pob)
bitstream=/work/shared/common/project_build/graphblas/graphlily-synthesize/spmv-syn/fp-pob/build_dir.hw/spmv.xclbin

# for ((i = 0; i < ${#GRAPH_DATASETS[@]}; i++)) do
#     ./benchmark $bitstream $GRAPH_DATASET_PATH/${GRAPH_DATASETS[i]}
# done

for ((i = 0; i < ${#NN_DATASETS[@]}; i++)) do
    ./benchmark $bitstream $NN_DATASET_PATH/${NN_DATASETS[i]}
done
