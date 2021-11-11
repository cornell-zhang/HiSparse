PROJ_ROOT = ..
IMPL := float

CXX := g++

CXXFLAGS += -Wall -O3 -std=c++14

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/cnpy/build -lcnpy

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/googletest/googletest/include
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/googletest/build/lib -lgtest

CXXFLAGS += -I/opt/xilinx/Xilinx_Vivado_vitis_2020.2/Vitis_HLS/2020.2/include
CXXFLAGS += -I$(PROJ_ROOT)/sw
CXXFLAGS += -I./include

LDFLAGS += -lrt -lstdc++

CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unused-function
CXXFLAGS += -Wno-unused-variable
CXXFLAGS += -Wno-misleading-indentation
CXXFLAGS += -Wno-int-in-bool-context
CXXFLAGS += -Wno-unused-label
CXXFLAGS += -Wno-sign-compare

performance_model: performance_model.cpp
	$(CXX) $(CXXFLAGS) $< -o performance_model $(LDFLAGS)

design_space_exp: design_space_exp.cpp
	$(CXX) $(CXXFLAGS) $< -o design_space_exp $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf performance_model
