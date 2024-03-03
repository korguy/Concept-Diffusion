import os
import yaml
import torch
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from pipelines import model_dict
from utils.data_util import load_data, load_scheduler
from utils.ptp_util import AttentionStore, register_attention_control

def generate(baselines, seeds, dataset, out_dir):
	for baseline in baselines:
		with open(baseline, 'r') as f:
			config = yaml.safe_load(f)
			_out_dir = f"{out_dir}/{config['name']}"
			os.makedirs(_out_dir, exist_ok=True)

			model = model_dict[config["model"]].from_pretrained(config["version"]).to("cuda")
			scheduler = load_scheduler(config["scheduler"], config["version"]) 
			data = load_data(dataset, config["model"])

			if config["model"] == "attend_and_excite":
				attention_store = AttentionStore()
				register_attention_control(model, attention_store)
				nouns = data[1]
				data = data[0]

			n = len(data) * len(seeds)

			print(f"Start Generating Images for {config['model']}")
			with tqdm(total=n) as pbar:
				for idx, _data in enumerate(data):
					prompt = _data
					
					if config["model"] == "attend_and_excite":
						token_indices = model.get_indices(prompt)
						indices_to_alter = []
						for n in nouns[idx]:
							try:
								indices_to_alter.append(token_indices[n])
							except:
								continue
					if config["model"] == "ours":
						prompt, nps = prompt

					__out_dir = f"{_out_dir}/{idx}_{prompt[:50]}"
					os.makedirs(__out_dir, exist_ok=True)

					for seed in seeds:
						try:
							if os.path.exists(f"{__out_dir}/{seed}.png"):
								continue
							g = torch.Generator("cuda").manual_seed(seed)
							
							if config["model"] == "attend_and_excite":
								image = model(
										prompt=prompt,
										attention_store=attention_store,
										indices_to_alter=indices_to_alter,
										generator=g,
										**config["params"]
									).images[0]
							elif config["model"] == "ours":
								image = model(
									prompt=prompt,
									noun_phrases=nps,
									generator=g,
									**config["params"]
								).images[0]
							else:
								image = model(
										prompt=prompt,
										generator=g,
										**config["params"]
									).images[0]
							image.save(f"{__out_dir}/{seed}.png")
							del g
						except:
							continue
						pbar.update(1)
			del model
			del scheduler

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("exp", type=str, help=".yaml file with experiment configuration")
	args = parser.parse_args()

	config = yaml.safe_load(open(args.exp, 'r'))

	out_dir = "run"
	os.makedirs(out_dir, exist_ok=True)

	out_dir = f"{out_dir}/{config['dataset'].split('/')[-1][:-4]}"
	os.makedirs(out_dir, exist_ok=True)

	generate(config["baselines"], config["seed"], config["dataset"], out_dir)
