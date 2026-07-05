
gen_bin:
	mkdir -p img_bin; rm -rf img_bin/*
	imgtobin.py -i mnist_1.jpg -o img_bin -c 4 -d 0,0,0,0 -m 0.1307 -sc 0.3081
	#imgtobin.py -i mnist_1.jpg -o img_bin -c 4 -d 0,0,0,0

train:
	python3 lenet.py -train

test_jpg:
	python3 lenet.py -test -j mnist_1.jpg

test_bin:
	python3 lenet.py -test -b img_bin/mnist_1.bin

export_onnx:
	python3 lenet.py -onnx model.onnx

test_onnx:
	python3 infer_onnx.py model.onnx img_bin/mnist_1.bin

all: test_jpg
