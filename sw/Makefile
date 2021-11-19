PROJ_ROOT = ..
IMPL := float_pob
ifeq ($(IMPL), float_pob)
	HW_DIR = $(PROJ_ROOT)/spmv-fp
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv-fp/libfpga
else ifeq ($(IMPL), float_stall)
	HW_DIR = $(PROJ_ROOT)/spmv-fp
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv-fp/libfpga
else
	HW_DIR = $(PROJ_ROOT)/spmv
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv/libfpga
endif

HOST_ARCH = x86

CXX := g++

CXXFLAGS += -Wall -std=c++11

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/cnpy/build -lcnpy

CXXFLAGS += -I$(PROJ_ROOT)/sw
CXXFLAGS += -I$(LIBFPGA_DIR)

include $(PROJ_ROOT)/xrt/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)

include $(PROJ_ROOT)/xrt/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

LDFLAGS += -lrt -lstdc++

CXXFLAGS += -Wno-maybe-uninitialized
CXXFLAGS += -Wno-uninitialized
CXXFLAGS += -Wno-int-in-bool-context

$(HW_DIR)/build_dir.hw_emu/spmv.xclbin:
	cd $(HW_DIR); make -j5 build TARGET=hw_emu IMPL=$(IMPL)

$(HW_DIR)/build_dir.hw/spmv.xclbin:
	cd $(HW_DIR); make -j5 build TARGET=hw IMPL=$(IMPL)

host: host.cpp
	$(CXX) $(CXXFLAGS) -g $(xcl2_SRCS) $< -o host $(LDFLAGS)

benchmark: benchmark.cpp $(HW_DIR)/build_dir.hw/spmv.xclbin
	$(CXX) $(CXXFLAGS) -O3 $(xcl2_SRCS) $< -o benchmark $(LDFLAGS)
	./bm.sh $(HW_DIR)/build_dir.hw/spmv.xclbin

measure_matrix: measure_matrix.cpp
	$(CXX) $(CXXFLAGS) $(xcl2_SRCS) $< -o measure_matrix $(LDFLAGS)
	./measure_matrix

.PHONY: hw_emu hw
hw_emu: host $(HW_DIR)/build_dir.hw_emu/spmv.xclbin
	cp $(HW_DIR)/emconfig.json .
	./host hw_emu $(HW_DIR)/build_dir.hw_emu/spmv.xclbin

hw: host $(HW_DIR)/build_dir.hw/spmv.xclbin
	./host hw $(HW_DIR)/build_dir.hw/spmv.xclbin

.PHONY: clean cleanall
clean:
	rm -rf emconfig.json
	rm -rf .run *.log *.csv

cleanall: clean
	rm -rf host benchmark
