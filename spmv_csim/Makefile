PROJ_ROOT = ..
IMPL := fixed

CXX := g++

CXXFLAGS += -Wall -O3 -std=c++14

CXXFLAGS += -I$(CNPY_INCLUDE)
LDFLAGS += -L$(CNPY_LIB) -lcnpy

CXXFLAGS += -I$(VITIS)/../../Vitis_HLS/2020.2/include
CXXFLAGS += -I$(PROJ_ROOT)/sw
CXXFLAGS += -I./csim_headers

LDFLAGS += -lrt -lstdc++

CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unused-function
CXXFLAGS += -Wno-unused-variable
CXXFLAGS += -Wno-misleading-indentation
CXXFLAGS += -Wno-int-in-bool-context
CXXFLAGS += -Wno-unused-label
CXXFLAGS += -Wno-sign-compare
CXXFLAGS += -Wno-maybe-uninitialized
CXXFLAGS += -Wno-uninitialized

ifeq ($(IMPL), float_pob)
	HW_DIR = $(PROJ_ROOT)/spmv-fp
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv-fp/libfpga
	CXXFLAGS += -DFP_POB
else ifeq ($(IMPL), float_stall)
	HW_DIR = $(PROJ_ROOT)/spmv-fp
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv-fp/libfpga
	CXXFLAGS += -DFP_STALL
else
	HW_DIR = $(PROJ_ROOT)/spmv
	LIBFPGA_DIR = $(PROJ_ROOT)/spmv/libfpga
endif

.PHONY: copy_files
copy_files:
	mkdir -p csim_headers
	cp $(HW_DIR)/k2k_relay.cpp csim_headers/k2k_relay.h
	cp $(HW_DIR)/spmv_result_drain.cpp csim_headers/spmv_result_drain.h
	cp $(HW_DIR)/spmv_vector_loader.cpp csim_headers/spmv_vector_loader.h
	cp $(HW_DIR)/spmv_sk0.cpp csim_headers/spmv_sk0.h
	cp $(HW_DIR)/spmv_sk1.cpp csim_headers/spmv_sk1.h
	cp $(HW_DIR)/spmv_sk2.cpp csim_headers/spmv_sk2.h
	cp $(LIBFPGA_DIR)/*.h csim_headers/

csim: csim.cpp copy_files
	$(CXX) $(CXXFLAGS) $< -o csim $(LDFLAGS)
	 ./csim

.PHONY: clean cleanall
clean:
	rm -rf csim

cleanall: clean
	rm -rf csim_headers
