
.PHONY: query_device mem_bandwidth

query_device:
	nvcc -std=c++17 -O2 -Xcompiler=-Wall,-Wextra cuda_query_device.cu -o cuda_query_device

mem_bandwidth:
	nvcc -std=c++17 -O3 -Xcompiler=-Wall,-Wextra cuda_mem_bandwidth.cu -o cuda_mem_bandwidth
