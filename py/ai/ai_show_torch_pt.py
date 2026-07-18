#!/usr/bin/env python3

import os, sys, argparse
import torch


def resolve_model_path(model_file):
	"""Resolve a model path from cwd or the script directory."""
	candidates = [
		model_file,
		os.path.abspath(model_file),
		os.path.join(os.getcwd(), model_file),
		os.path.join(os.path.dirname(__file__), model_file),
	]
	for candidate in candidates:
		if os.path.exists(candidate):
			return candidate
	return model_file

def get_input_info(ep):
	"""Recursively collect input shapes/dtypes/devices from ExportedProgram."""
	inputs = []

	def walk(value):
		if isinstance(value, torch.Tensor):
			inputs.append({
				'shape': tuple(value.shape),
				'dtype': value.dtype,
				'device': value.device,
			})
		elif isinstance(value, (tuple, list)):
			for item in value:
				walk(item)
		elif isinstance(value, dict):
			for item in value.values():
				walk(item)

	for arg in getattr(ep, 'example_inputs', []):
		walk(arg)
	return inputs


def show_model(args):
	# Load the pytorch model
	model_file = resolve_model_path(args.model)
	loaded_ep = torch.export.load(model_file)

	print("===== Graph Code =====")
	print(loaded_ep.graph_module.code)
	print("nodes num in graph: ", len(loaded_ep.graph.nodes))

	print("\n===== Weights =====")
	for k, v in loaded_ep.state_dict.items():
		print(k, v.shape)

	print("\n===== IOs =====")
	input_info = get_input_info(loaded_ep)
	for i, info in enumerate(input_info):
		print(f"input {i}: shape={info['shape']}, dtype={info['dtype']}, device={info['device']}")

	if args.v:
		print("="*60)
		print(loaded_ep)


def init_param(args):
	parser = argparse.ArgumentParser(description="Show Pytorch pt model by torch.export.export()")

	parser.add_argument("-m", "-model", "--model", dest="model", type=str, required=True, default="model.pt",
		help="ckpt model file")

	parser.add_argument("-v", action='store_true', required=False,
		help="Show verbose info")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	show_model(args)
