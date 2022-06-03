#!/bin/bash
# usage: `./generate_hw_ini DDR` or `./generate_hw_ini HBM <number of channels>`

hw_config=spmspv.ini

cat <<EOF > $hw_config
[connectivity]
slr=spmspv_1:SLR0
sp=spmspv_1.vector:HBM[30]
sp=spmspv_1.result:HBM[31]
EOF

if [ $1 = "DDR" ];then
cat <<EOF >> spmspv.ini
sp=spmspv_1.mat_0:DDR[0]
sp=spmspv_1.mat_indptr_0:DDR[0]
sp=spmspv_1.mat_partptr_0:DDR[0]
EOF
elif [ "$1" = "HBM" ];then
  for (( i = 0; i < $2; i++ )); do
    echo "sp=spmspv_1.mat_$i:HBM[$i]" >> $hw_config
    echo "sp=spmspv_1.mat_indptr_$i:HBM[$i]" >> $hw_config
    echo "sp=spmspv_1.mat_partptr_$i:HBM[$i]" >> $hw_config
  done
fi
