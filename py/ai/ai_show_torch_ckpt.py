#!/usr/bin/env python3

import os, sys, argparse
import torch

def show_model(args):
	# Load the pytorch model
	models = torch.load(args.model, map_location='cpu')

	print("stats_dict main key list:", list(models))

	if args.exportw:
		out_folder = args.exportw
		os.makedirs(out_folder, exist_ok=True)
		print("export-weights out folder:", out_folder)

	if args.key:
		print("stats_dict sub key list:", list(models[args.key]))
		for key,value in models[args.key].items():
				print("subkey:", key)

				## export weights
				if args.exportw:
					out_fn = os.path.join(out_folder, key)
					torch.save(value, out_fn)

	if args.v:
		print(models)


def init_param(args):
	parser = argparse.ArgumentParser(description="Show Pytorch Checkpoint")

	parser.add_argument("-model", type=str, required=True, default="model.pth",
		help="ckpt model file")
	parser.add_argument("-key", type=str, required=False,
		help="key name in stats_dict")

	parser.add_argument("-v", action='store_true', required=False,
		help="Show verbose info")

	parser.add_argument("-exportw", type=str, required=False,
		help="export weights to output folder")

	return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	show_model(args)
