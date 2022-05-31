PROJ_ROOT = ..

TARGET := sw_emu

DEVICE = /opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm

######### host program configuration #########
HOST_ARCH = x86

CXX := g++

CXXFLAGS += -Wall -std=c++14

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/cnpy/build -lcnpy

HW_DIR = .
LIBFPGA_DIR = ./libfpga

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
CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unused-function

######### xcel bitstream configuration #########

HLS_DIR := ./_x.$(TARGET)
IMP_DIR := ./build_dir.$(TARGET)

VPP := v++

CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps
CLFLAGS += -I./libfpga

ifneq ($(TARGET), hw)
	CLFLAGS += -g
else
	LDCLFLAGS += --optimize 3 --kernel_frequency 300
endif

SPMSPV_XCLBIN = $(IMP_DIR)/spmspv.xclbin

######### build bitstream #########

emconfig.json:
	emconfigutil --platform $(DEVICE)

build: $(SPMSPV_XCLBIN) emconfig.json

LDCLFLAGS += --config spmspv.ini
KERNEL_OBJS += $(HLS_DIR)/spmspv.xo

$(HLS_DIR)/spmspv.xo: spmspv.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmspv -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/spmspv.xo

$(SPMSPV_XCLBIN): $(KERNEL_OBJS)
	mkdir -p $(IMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(IMP_DIR) -l $(LDCLFLAGS) -o'$@' $^

######### build host #########
host: host.cpp
	$(CXX) $(CXXFLAGS) -g $(xcl2_SRCS) $< -o host $(LDFLAGS)

######### build benchmark #########
benchmark: benchmark.cpp
	$(CXX) $(CXXFLAGS) -g $(xcl2_SRCS) $< -o benchmark $(LDFLAGS)

######### run test #########
check: build host
	./host $(TARGET) $(SPMSPV_XCLBIN)

######### run test #########
bench: build benchmark
	./benchmark $(SPMSPV_XCLBIN)

######### clean stuff #########
.PHONY: clean
clean:
	rm -rf *.log
	rm -rf .run

.PHONY: cleanall
cleanall: clean
	rm -rf ./host
	rm -rf ./benchmark
	rm -rf ./emconfig.json
	rm -rf ./.Xil
	rm -rf ./_x.sw_emu ./build_dir.sw_emu
	rm -rf ./_x.hw_emu ./build_dir.hw_emu
	rm -rf ./_x.hw ./build_dir.hw
