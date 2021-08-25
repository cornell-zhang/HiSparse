TARGET := hw_emu
UNIT := pe_tb
IMPL := float

DEVICE = /opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm

HLS_DIR := ./$(UNIT)_x.$(TARGET)
IMP_DIR := ./$(UNIT)_build_dir.$(TARGET)

VPP := v++

CLFLAGS += -t $(TARGET) -g --platform $(DEVICE) --save-temps
ifeq ($(IMPL), float)
	CLFLAGS += -I../spmv-fp/libfpga
else
	CLFLAGS += -I../spmv/libfpga
endif

emconfig.json:
	emconfigutil --platform $(DEVICE)

.PHONY: build
build: emconfig.json $(IMP_DIR)/$(UNIT).xclbin

$(IMP_DIR)/$(UNIT).xclbin: $(UNIT).cpp $(UNIT).h $(UNIT).ini
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k $(UNIT) -I'$(<D)' -o $(HLS_DIR)/$(UNIT).xo $<
	$(VPP) $(CLFLAGS) --temp_dir $(IMP_DIR) --config $(UNIT).ini -l $(LDCLFLAGS) -o'$@' $(HLS_DIR)/$(UNIT).xo

.PHONY: clean
clean:
	rm -rf *.log
	rm -rf .run

.PHONY: cleanall
cleanall: clean
	rm -rf emconfig.json
	rm -rf *_x.* *_build_dir.*
