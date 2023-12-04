import os
import yaml
import torch
import numpy as np

from diffusers import PNDMScheduler

from pipelines import *

def generate(model, prompt, subjects, params, seed, out_dir):
	subjects = [x.strip() for x in subjects.split(',')]
	g = torch.Generator('cuda').manual_seed(seed)
	image = model(
			prompt = prompt,
			subjects = subjects,
			generator=g,
			**params).images[0]
	image.save(f"{out_dir}/{seed}_{prompt}.png")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('prompt', type=str, help='input prompt')
	parser.add_argument('subjects', type=str, help='list of subjects in input prompt')
	parser.add_argument('--seed', type=int, help='random seed')
	args = parser.parse_args()

	config = yaml.safe_load(open("configs/default.yaml", "r"))

	seed = np.random.randint(0, 100000000)
	if args.seed is not None:
		seed = int(args.seed)

	out_dir = "run"
	os.makedirs(out_dir, exist_ok=True)

	scheduler = PNDMScheduler.from_pretrained(config['version'], subfolder="scheduler")
	model = ConceptDiffusionPipeline.from_pretrained(config['version'], scheduler=scheduler)
	model = model.to('cuda')

	generate(model, args.prompt, args.subjects, config['params'], seed, out_dir)