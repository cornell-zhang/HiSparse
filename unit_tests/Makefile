PROJ_ROOT = ..
LIBFPGA_DIR = $(PROJ_ROOT)/spmv/libfpga
TB_DIR = $(PROJ_ROOT)/unit_test_wrapper

HOST_ARCH = x86

CXX := g++

CXXFLAGS += -Wall -g -std=c++11

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/cnpy/build -lcnpy

CXXFLAGS += -I$(TB_DIR)
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

emconfig.json:
	cp $(TB_DIR)/emconfig.json .

test_pe: test_pe.cpp $(xcl2_SRCS) emconfig.json
	$(CXX) $(CXXFLAGS) test_pe.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)

.PHONY: clean cleanall
clean:
	rm -rf .run *.log *.csv

cleanall:
	rm -rf emconfig.json
	rm -rf test_pe
