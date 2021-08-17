TARGET := hw_emu

DEVICE = /opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm

HLS_DIR := ./_x.$(TARGET)
IMP_DIR := ./build_dir.$(TARGET)

VPP := v++

CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps
CLFLAGS += -I./libfpga

ifneq ($(TARGET), hw)
	CLFLAGS += -g
else
	LDCLFLAGS += --optimize 3
endif

SPMV_XCLBIN = $(IMP_DIR)/spmv.xclbin

emconfig.json:
	emconfigutil --platform $(DEVICE)

build: $(SPMV_XCLBIN) emconfig.json

LDCLFLAGS += --config spmv.ini
KERNEL_OBJS += $(HLS_DIR)/spmv_sk0.xo

$(HLS_DIR)/spmv_sk0.xo: spmv_sk0.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmv_sk0 -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/spmv_sk1.xo

$(HLS_DIR)/spmv_sk1.xo: spmv_sk1.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmv_sk1 -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/spmv_sk2.xo

$(HLS_DIR)/spmv_sk2.xo: spmv_sk2.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmv_sk2 -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/spmv_vector_loader.xo

$(HLS_DIR)/spmv_vector_loader.xo: spmv_vector_loader.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmv_vector_loader -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/spmv_result_drain.xo

$(HLS_DIR)/spmv_result_drain.xo: spmv_result_drain.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k spmv_result_drain -I'$(<D)' -o'$@' $^

KERNEL_OBJS += $(HLS_DIR)/k2k_relay.xo

$(HLS_DIR)/k2k_relay.xo: k2k_relay.cpp
	mkdir -p $(HLS_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(HLS_DIR) -c -k k2k_relay -I'$(<D)' -o'$@' $^

$(SPMV_XCLBIN): $(KERNEL_OBJS)
	mkdir -p $(IMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(IMP_DIR) -l $(LDCLFLAGS) -o'$@' $^


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
