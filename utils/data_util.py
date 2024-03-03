import spacy
import pandas as pd
import numpy as np

from diffusers import LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, DDPMScheduler, LMSDiscreteScheduler

def load_data(path, method, return_raw_prompts=False):
	try:
		data = pd.read_csv(path)
	except pd.errors.ParserError:
		data = pd.read_csv(path, delimiter="|")
	if return_raw_prompts or method == "stable_diffusion" or method == "structure_diffusion" or method == "syntax_guided_generation":
		return data["prompt"]
	elif method == "composable_diffusion":
		prompts = []
		nlp = spacy.load("en_core_web_sm")
		for p in data["prompt"]:
			doc = nlp(p)
			noun_phrases = [chunk.text for chunk in doc.noun_chunks]
			if len(noun_phrases) != 0:
				prompts.append("|".join(noun_phrases))
		del nlp
		return prompts
	elif method == "attend_and_excite":
		nouns = []
		nlp = spacy.load("en_core_web_sm")
		for p in data["prompt"]:
			doc = nlp(p)
			_nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
			nouns.append(_nouns)
		del nlp
		return (data["prompt"], nouns)
	elif method == "ours":
		prompts = []
		nlp = spacy.load("en_core_web_sm")
		for p in data["prompt"]:
			doc = nlp(p)
			noun_phrases = [chunk.text for chunk in doc.noun_chunks]
			prompts.append((p, noun_phrases))
		del nlp
		return prompts

def load_scheduler(name, version):
	if name == "ddim":
		return DDIMScheduler.from_pretrained(version, subfolder="scheduler")
	elif name == "pndm":
		return PNDMScheduler.from_pretrained(version, subfolder="scheduler")
	elif name == "ddpm":
		return DDPMScheduler.from_pretrained(version, subfolder="scheduler")
	elif name == "lms":
		return LMSDiscreteScheduler.from_pretrained(version, subfolder="scheduler")
	else:
		raise NotImplementedError("Unsupported Scheduler")
