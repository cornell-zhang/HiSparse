#!/bin/bash

DATASET=../datasets/graph/gplus_108K_13M_csr_float32.npz
BUILD_DIR=/work/shared/common/project_build/spmspv_imp
XCLBIN=build_dir.hw/spmspv.xclbin
mkdir -p ./bmlogs
# ./benchmark_ddr gplus_ddr $DATASET ${BUILD_DIR}/ddr/${XCLBIN} 1 "./bmlogs/gplus_ddr.log"
# ./benchmark gplus_2ch $DATASET ${BUILD_DIR}/2ch/${XCLBIN} 2 "./bmlogs/gplus_2ch.log"
# ./benchmark gplus_4ch $DATASET ${BUILD_DIR}/4ch/${XCLBIN} 4 "./bmlogs/gplus_4ch.log"
# ./benchmark gplus_6ch $DATASET ${BUILD_DIR}/6ch/${XCLBIN} 6 "./bmlogs/gplus_6ch.log"
# ./benchmark gplus_8ch $DATASET ${BUILD_DIR}/8ch_opt/${XCLBIN} 8 "./bmlogs/gplus_8ch.log"
./benchmark gplus_8ch $DATASET ./${XCLBIN} 8 "./bmlogs/gplus_8ch.log"
# ./benchmark gplus_10ch $DATASET ${BUILD_DIR}/10ch/${XCLBIN} 10 "./bmlogs/gplus_10ch.log"
