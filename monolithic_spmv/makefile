PROJ_ROOT = ..

TARGET := sw_emu

DEVICE = /opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm

######### host program configuration #########
HOST_ARCH = x86

CXX := g++

CXXFLAGS += -Wall -std=c++11

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

######### xcel bitstream configuration #########

HLS_DIR := ./_x.$(TARGET)
IMP_DIR := ./build_dir.$(TARGET)

VPP := v++

CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps
CLFLAGS += -I./libfpga

ifneq ($(TARGET), hw)
	CLFLAGS += -g
else
	LDCLFLAGS += --optimize 3 --kernel_frequency 200
endif

SPMV_XCLBIN = $(IMP_DIR)/spmv.xclbin

######### build bitstream #########

emconfig.json:
	emconfigutil --platform $(DEVICE)

build: $(SPMV_XCLBIN) emconfig.json

LDCLFLAGS += --config spmv.ini
KERNEL_OBJS += $(HLS_DIR)/spmv.xo

$(HLS_DIR)/spmv.xo: spmv.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmv -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/spmv.xo

$(SPMV_XCLBIN): $(KERNEL_OBJS)
	mkdir -p $(IMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(IMP_DIR) -l $(LDCLFLAGS) -o'$@' $^

######### build host #########
host: host.cpp
	$(CXX) $(CXXFLAGS) -g $(xcl2_SRCS) $< -o host $(LDFLAGS)

######### run test #########
check: build host
	./host $(TARGET) $(SPMV_XCLBIN)

######### clean stuff #########
.PHONY: clean
clean:
	rm -rf *.log
	rm -rf .run

.PHONY: cleanall
cleanall: clean
	rm -rf emconfig.json
	rm -rf ./_x.sw_emu ./build_dir.sw_emu
	rm -rf ./_x.hw_emu ./build_dir.hw_emu
	rm -rf ./_x.hw ./build_dir.hw
