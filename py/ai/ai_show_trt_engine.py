#!/usr/bin/env python3
# Show TensorRT engine (*.plan / *.engine) input/output tensor shape information.
# Example:
#   python3 ai_show_trt_engine.py -model model.plan
#   python3 ai_show_trt_engine.py -model model.plan -d 1
import sys, argparse, ctypes, traceback

try:
	import tensorrt as trt
except ImportError as e:
	print("Error: failed to import TensorRT:", e)
	print("Please install TensorRT or run this script inside the TensorRT environment.")
	sys.exit(1)


log_levels = {
	"internal_error": trt.Logger.INTERNAL_ERROR,
	"error": trt.Logger.ERROR,
	"warning": trt.Logger.WARNING,
	"info": trt.Logger.INFO,
	"verbose": trt.Logger.VERBOSE,
}


def dims2tuple(dims):
	return tuple(int(d) for d in dims)


def has_dynamic_dim(shape):
	return any(d < 0 for d in shape)


def safe_call(fn, default=None):
	try:
		return fn()
	except Exception:
		return default


def engine_attr(engine, name, default=None):
	try:
		value = getattr(engine, name)
		return value() if callable(value) else value
	except Exception:
		return default


def load_engine(model, logger, plugins):
	# Load extra custom plugin libraries before deserializing the engine
	for lib in plugins:
		ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
		print("Plugin library:", lib)

	# Load the standard TensorRT plugin libraries (nvinfer_plugin, etc.)
	safe_call(lambda: trt.init_libnvinfer_plugins(logger, ""))

	runtime = trt.Runtime(logger)
	with open(model, "rb") as f:
		data = f.read()

	print("File size:", len(data), "bytes")

	engine = runtime.deserialize_cuda_engine(data)
	if engine is None:
		raise RuntimeError(
			"failed to deserialize the engine!\n"
			"  - engine built with a different TensorRT version (current: %s)\n"
			"  - engine built for a different GPU / platform / compute capability\n"
			"  - required custom plugin libraries are missing (load them with -plugin)\n"
			"  Please rebuild the engine on this machine." % trt.__version__)

	if not hasattr(engine, "num_io_tensors"):
		raise RuntimeError("this script requires TensorRT >= 10.x (ICudaEngine tensor API)")

	return engine


def get_io_tensors(engine):
	inputs, outputs = [], []
	for i in range(engine.num_io_tensors):
		name = engine.get_tensor_name(i)
		mode = engine.get_tensor_mode(name)
		if mode == trt.TensorIOMode.INPUT:
			inputs.append(name)
		elif mode == trt.TensorIOMode.OUTPUT:
			outputs.append(name)
	return inputs, outputs


def describe_tensor(engine, name, detail):
	shape = dims2tuple(engine.get_tensor_shape(name))
	shape_str = str(shape)
	if has_dynamic_dim(shape):
		shape_str += " (dynamic)"

	dtype = engine.get_tensor_dtype(name).name
	desc = "shape=%s  dtype=%s" % (shape_str, dtype)

	if detail >= 1:
		loc = safe_call(lambda: engine.get_tensor_location(name).name)
		if loc:
			desc += "  loc=%s" % loc
		fmt = safe_call(lambda: engine.get_tensor_format_desc(name))
		if fmt:
			desc += "  format=%s" % fmt

	if safe_call(lambda: engine.is_shape_inference_io(name), False):
		desc += "  (shape tensor)"
	return desc


def profile_shapes(engine, name, profile):
	# Some TensorRT builds return malformed (empty) Dims for output tensors
	rank = len(engine.get_tensor_shape(name))
	try:
		shapes = engine.get_tensor_profile_shape(name, profile)
		ret = [dims2tuple(s) for s in shapes]
	except Exception:
		return None
	if len(ret) != 3 or any(len(s) != rank for s in ret):
		return None
	return ret


def show_profiles(engine, inputs, outputs):
	profile_num = safe_call(lambda: engine.num_optimization_profiles, 0)
	if not profile_num:
		return

	dynamic = False
	for name in inputs + outputs:
		if has_dynamic_dim(dims2tuple(engine.get_tensor_shape(name))):
			dynamic = True
			break

	blocks = []
	for p in range(profile_num):
		rows = []
		for kind, names in (("input", inputs), ("output", outputs)):
			for name in names:
				shapes = profile_shapes(engine, name, p)
				if not shapes or len(shapes) != 3:
					continue
				mn, opt, mx = shapes
				if mn != opt or opt != mx:
					dynamic = True
				rows.append("  %-7s %s  min=%s  opt=%s  max=%s" % (kind + ":", name, mn, opt, mx))
		if rows:
			if profile_num > 1:
				rows.insert(0, "Profile %d:" % p)
			blocks.extend(rows)

	# Static engines already show the exact shapes in the IO section
	if not dynamic or not blocks:
		return

	print("\n------ Optimization Profiles  ------")
	print("\n".join(blocks))


def show_layers(engine, detail):
	inspector = safe_call(lambda: engine.create_engine_inspector())
	if inspector is None:
		print("\n(engine inspector is not available)")
		return

	if detail == 1:
		info = safe_call(lambda: inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))
		if info:
			# Drop the I/O tensor section, it is printed below
			info = info.split("\nI/O Tensors:")[0]
	else:
		info = safe_call(lambda: inspector.get_engine_information(trt.LayerInformationFormat.JSON))

	if not info:
		print("\n(engine inspector returned no information)")
		return

	print("\n------ Engine Layers  ------")
	print(info)


def show_engine(args, logger):
	print("Engine: ", args.model)
	engine = load_engine(args.model, logger, args.plugin or [])

	print("TensorRT version:", trt.__version__)
	print("Engine name:", engine_attr(engine, "name", "?"))
	print("Layer num:", engine_attr(engine, "num_layers", "?"))
	print("Optimization profile num:", engine_attr(engine, "num_optimization_profiles", "?"))

	mem = engine_attr(engine, "device_memory_size_v2")
	if mem is None:
		mem = engine_attr(engine, "device_memory_size")
	if isinstance(mem, int) and mem >= 0:
		print("Device memory: %d bytes (%.2f MiB)" % (mem, mem / 1024.0 / 1024.0))

	if args.d >= 1:
		show_layers(engine, args.d)

	inputs, outputs = get_io_tensors(engine)

	print("\n------ Engine Primary IO  ------")
	print("Input num:", len(inputs))
	for name in inputs:
		print("  input: ", name, describe_tensor(engine, name, args.d))
	print("Output num:", len(outputs))
	for name in outputs:
		print("  output:", name, describe_tensor(engine, name, args.d))

	show_profiles(engine, inputs, outputs)


def init_param(args):
	parser = argparse.ArgumentParser(description="Show TensorRT Engine (*.plan) Input/Output Information")

	parser.add_argument("-model", type=str, required=True, default="model.plan",
		help="TensorRT engine file, e.g. model.plan / model.engine")

	parser.add_argument("-d", type=int, required=False, default=0,
		help="Show detail level. 0: IO shapes, 1: + layer list, 2: + full inspector JSON. [0, 2]")

	parser.add_argument("-loglevel", type=str, required=False, default="warning",
		choices=list(log_levels.keys()),
		help="TensorRT logger severity. [warning]")

	parser.add_argument("-plugin", type=str, required=False, action="append",
		help="extra plugin library (*.so) to load before deserializing, can be repeated")

	return parser.parse_args(args)


if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	logger = trt.Logger(log_levels[args.loglevel])
	try:
		show_engine(args, logger)
	except FileNotFoundError:
		print("Error: engine file not found:", args.model)
		sys.exit(1)
	except Exception as e:
		if args.d >= 2:
			traceback.print_exc()
		print("Error:", e)
		sys.exit(1)
