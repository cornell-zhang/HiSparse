#!/bin/bash

GRAPH_DATASET_PATH=../datasets/graph
GRAPH_DATASETS=(gplus_108K_13M_csr_float32.npz
                ogbl_ppa_576K_42M_csr_float32.npz
                hollywood_1M_113M_csr_float32.npz
                pokec_1633K_31M_csr_float32.npz
                ogbn_products_2M_124M_csr_float32.npz
                mouse_gene_45K_29M_csr_float32.npz)

NN_DATASET_PATH=../datasets/pruned_nn
NN_DATASETS=(transformer_50_512_33288_csr_float32.npz
             transformer_60_512_33288_csr_float32.npz
             transformer_70_512_33288_csr_float32.npz
             transformer_80_512_33288_csr_float32.npz
             transformer_90_512_33288_csr_float32.npz
             transformer_95_512_33288_csr_float32.npz)

bitstream=$1
impl=$2
vb=4
if [ $impl = "float_pob" ]
then
    ob=1
else
    ob=8
fi

for ((i = 0; i < ${#GRAPH_DATASETS[@]}; i++)) do
    ./benchmark $bitstream $GRAPH_DATASET_PATH/${GRAPH_DATASETS[i]} $vb $ob
done

for ((i = 0; i < ${#NN_DATASETS[@]}; i++)) do
    ./benchmark $bitstream $NN_DATASET_PATH/${NN_DATASETS[i]} $vb $ob
done
