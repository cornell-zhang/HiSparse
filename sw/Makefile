PROJ_ROOT = ..
IMPL := fixed
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

CXXFLAGS += -I$(CNPY_INCLUDE)
LDFLAGS += -L$(CNPY_LIB) -lcnpy

CXXFLAGS += -I$(PROJ_ROOT)/sw
CXXFLAGS += -I$(LIBFPGA_DIR)
CXXFLAGS += -I$(HLS_INCLUDE)

.PHONY: check_hls_include
check_hls_include:
ifndef HLS_INCLUDE
	$(error HLS_INCLUDE is not set, please set it to the path to ap_fixed.h)
endif

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

.PHONY: demo host hw_emu hw benchmark
host: host.cpp
	$(CXX) $(CXXFLAGS) -g $(xcl2_SRCS) $< -o host $(LDFLAGS)

demo: benchmark.cpp ../demo_spmv.xclbin
	$(CXX) $(CXXFLAGS) -O3 $(xcl2_SRCS) $< -o benchmark $(LDFLAGS)
	./bm.sh ../demo_spmv.xclbin fixed

benchmark: benchmark.cpp $(HW_DIR)/build_dir.hw/spmv.xclbin
	$(CXX) $(CXXFLAGS) -O3 $(xcl2_SRCS) $< -o benchmark $(LDFLAGS)
	./bm.sh $(HW_DIR)/build_dir.hw/spmv.xclbin $(IMPL)

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
