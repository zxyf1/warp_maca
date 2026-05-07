# Build the pure CUDA C++ reproduction.
#
# For MetaX C500, replace `nvcc` with the MetaX compiler command if different.
# The GPU arch flag (-arch) may also need adjusting:
#   NVIDIA A100  → -arch=sm_80
#   NVIDIA V100  → -arch=sm_70
#   MetaX C500   → check MetaX documentation for the equivalent flag

NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS ?= -O2 -g -arch=$(ARCH)
TARGET     = cell_lookup

all: $(TARGET)

$(TARGET): cell_lookup.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
