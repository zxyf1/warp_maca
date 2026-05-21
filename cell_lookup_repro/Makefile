# Build standalone CUDA reproductions of the MetaX predicated vec3 write bug.
#
# For MetaX C500, replace `nvcc` with the MetaX compiler command if different.
# The GPU arch flag (-arch) may also need adjusting:
#   NVIDIA A100  → -arch=sm_80
#   NVIDIA V100  → -arch=sm_70
#   MetaX C500   → check MetaX documentation for the equivalent flag
#
# ── Targets ───────────────────────────────────────────────────────────────────
#  cell_lookup   : nvcc offline compilation (bug does NOT reproduce here)
#  repro_nvrtc   : nvcc host + NVRTC JIT (bug reproduces on MetaX via NVRTC)
#
# ── Usage (repro_nvrtc) ───────────────────────────────────────────────────────
#  make repro_nvrtc
#  ./repro_nvrtc               # run repro (no arguments needed)
#  ./repro_nvrtc --dump-bc     # also dump cell_lookup.bc (NVIDIA only)

NVCC          ?= nvcc
ARCH          ?= sm_80
NVCCFLAGS     ?= -O2 -g -arch=$(ARCH)

TARGET        = cell_lookup
NVRTC_TARGET  = repro_nvrtc

all: $(TARGET)

$(TARGET): cell_lookup.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# repro_nvrtc: host code compiled offline with nvcc.
# The device kernels in KERNEL_SRC[] are JIT-compiled at runtime via NVRTC.
$(NVRTC_TARGET): repro_nvrtc.cu
	$(NVCC) $(NVCCFLAGS) repro_nvrtc.cu -lcuda -lnvrtc -o $@

run: $(TARGET)
	./$(TARGET)

run_nvrtc: $(NVRTC_TARGET)
	./$(NVRTC_TARGET)

run_nvrtc_dump_bc: $(NVRTC_TARGET)
	./$(NVRTC_TARGET) --dump-bc

clean:
	rm -f $(TARGET) $(NVRTC_TARGET) cell_lookup.bc

.PHONY: all run run_nvrtc run_nvrtc_dump_bc clean
