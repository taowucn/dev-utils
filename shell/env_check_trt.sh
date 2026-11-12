echo "===== TensorRT Packages ====="
dpkg-query -W 'tensorrt*' 'libnvinfer*' 2>/dev/null

echo "===== Python TensorRT ====="
python3 -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null

echo "===== TensorRT Header ====="
grep NV_TENSORRT /usr/include/aarch64-linux-gnu/NvInferVersion.h 2>/dev/null

echo "===== TensorRT Runtime ====="
ldconfig -p | grep libnvinfer.so

