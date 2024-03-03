from typing import Any, Callable, Dict, List, Optional, Union

import torch
import warnings
warnings.filterwarnings("ignore")
import torch.distributions as dist
import numpy as np

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

def get_perpendicular_component(x, y):
	assert x.shape == y.shape
	return x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y

def get_overlapping_component(x, y):
	assert x.shape == y.shape
	return ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y

def extract_concept(score, th=0.9):
	tmp = torch.quantile(torch.abs(score).flatten(start_dim=2),
								th,
								dim=2,
								keepdim=False
							)
	score = torch.where(
		torch.abs(score) >= tmp[:, :, None, None],
		score,
		torch.zeros_like(score)
	)
	return score

class OurPipeline(StableDiffusionPipeline):
	def _encode_prompt(
		self,
		prompt,
		device,
		num_images_per_prompt,
		do_classifier_free_guidance,
		negative_prompt=None,
		prompt_embeds: Optional[torch.FloatTensor] = None,
		negative_prompt_embeds: Optional[torch.FloatTensor] = None,
	):
		if prompt is not None and isinstance(prompt, str):
			batch_size = 1
		elif prompt is not None and isinstance(prompt, list):
			batch_size = len(prompt)
		else:
			batch_size = prompt_embeds.shape[0]

		if prompt_embeds is None:		
			text_inputs = self.tokenizer(
				prompt,
				padding="max_length",
				max_length=self.tokenizer.model_max_length,
				truncation=True,
				return_tensors="pt",
			)
			text_input_ids = text_inputs.input_ids
			untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

			if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
				text_input_ids, untruncated_ids
			):
				removed_text = self.tokenizer.batch_decode(
					untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
				)
				logger.warning(
					"The following part of your input was truncated because CLIP can only handle sequences up to"
					f" {self.tokenizer.model_max_length} tokens: {removed_text}"
				)

			if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
				attention_mask = text_inputs.attention_mask.to(device)
			else:
				attention_mask = None

			attention_mask = None
			# attention_mask = text_inputs.attention_mask.to(device)

			ids = text_inputs.input_ids[-1]
			tokens = torch.masked_select(ids, text_inputs.attention_mask[-1].bool())
			tokens = self.tokenizer.decode(tokens)

			prompt_embeds = self.text_encoder(
				text_input_ids.to(device),
				attention_mask=attention_mask,
			)
			prompt_embeds = prompt_embeds[0]

		prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

		bs_embed, seq_len, _ = prompt_embeds.shape
		# duplicate text embeddings for each generation per prompt, using mps friendly method
		prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
		prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

		# get unconditional embeddings for classifier free guidance
		if do_classifier_free_guidance and negative_prompt_embeds is None:
			uncond_tokens: List[str]
			if negative_prompt is None:
				uncond_tokens = [""] * batch_size
			elif type(prompt) is not type(negative_prompt):
				raise TypeError(
					f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
					f" {type(prompt)}."
				)
			elif isinstance(negative_prompt, str):
				uncond_tokens = [negative_prompt]
			elif batch_size != len(negative_prompt):
				raise ValueError(
					f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
					f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
					" the batch size of `prompt`."
				)
			else:
				uncond_tokens = negative_prompt

			# textual inversion: procecss multi-vector tokens if necessary
			if isinstance(self, TextualInversionLoaderMixin):
				uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

			max_length = prompt_embeds.shape[1]
			uncond_input = self.tokenizer(
				uncond_tokens,
				padding="max_length",
				max_length=max_length,
				truncation=True,
				return_tensors="pt",
			)

			if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
				attention_mask = uncond_input.attention_mask.to(device)
			else:
				attention_mask = None
			# attention_mask = uncond_input.attention_mask.to(device)

			negative_prompt_embeds = self.text_encoder(
				uncond_input.input_ids.to(device),
				attention_mask=attention_mask,
			)
			negative_prompt_embeds = negative_prompt_embeds[0]

		if do_classifier_free_guidance:
			# duplicate unconditional embeddings for each generation per prompt, using mps friendly method
			seq_len = negative_prompt_embeds.shape[1]

			negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

			negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
			negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

			# For classifier free guidance, we need to do two forward passes.
			# Here we concatenate the unconditional and text embeddings into a single batch
			# to avoid doing two forward passes
			prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

		return tokens, prompt_embeds

	@staticmethod
	def _update_latent(latents, loss, step_size):
		grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
		latents = latents - step_size * grad_cond
		return latents

	def _ours_step(self, latent_model_input, prompt_embeds, t, i):
		noise_pred = []
		for j in range(prompt_embeds.size(0)):
			latent = latent_model_input.clone()
			_noise_pred = self.unet(
				latent,t,encoder_hidden_states=prompt_embeds[j][None]
				)[0]
			noise_pred.append(_noise_pred)
		noise_pred = torch.cat(noise_pred, dim=0)
		return noise_pred

	def plot_similarities(self, name):
		import matplotlib.pyplot as plt
		import seaborn as sns

		cosine = np.array(self.sim).transpose()

		sns.set_style('darkgrid')
		plt.rc('axes', titlesize=18)     # fontsize of the axes title
		plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
		plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
		plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
		plt.rc('legend', fontsize=13)    # legend fontsize
		plt.rc('font', size=13)

		plt.figure(figsize=(8,8), tight_layout=True)
		for i in range(len(self.nps)+1):
			plt.plot(list(range(cosine.shape[1])), cosine[i], 'o-', color=sns.color_palette('pastel')[i+1])

		plt.legend(
			title='Concept', title_fontsize = 13,
			labels=self.nps + ["abstract"]
		)
		plt.xlabel("Timesteps")
		plt.ylabel("Cosine Similarity")
		plt.savefig(name)


	@torch.no_grad()
	def __call__(
		self,
		prompt: str,
		noun_phrases: List[str], 
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
		concept_guidance_scale: float = 5.5,
		negative_prompt: Optional[Union[str, List[str]]] = None,
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		latents: Optional[torch.FloatTensor] = None,
		prompt_embeds: Optional[torch.FloatTensor] = None,
		negative_prompt_embeds: Optional[torch.FloatTensor] = None,
		output_type: Optional[str] = "pil",
		return_dict: bool = True,
		callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: int = 1,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		edit_momentum_scale=0.7,
		edit_mom_beta=0.2,
		subject_th = 0.5,
		abstract_th = 0.5,
		warmup = 10,
		cooldown=20,
		window = 1,
		subject_percentile = 0.5,
		**kwargs
	):
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor

		self.check_inputs(
			prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
		)
		
		batch_size = 1
		device = self._execution_device

		do_classifier_free_guidance = guidance_scale > 1.0

		prompt = [""] + noun_phrases + [prompt]

		tokens, prompt_embeds = self._encode_prompt(
			prompt,
			device,
			num_images_per_prompt,
			False,
			negative_prompt,
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_prompt_embeds,
		) # prompt_embeds: [N, 77, 768]

		self.tokens = tokens
		self.nps = noun_phrases 

		self.sim =[]

		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps

		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents(
			batch_size * num_images_per_prompt,
			num_channels_latents,
			height,
			width,
			prompt_embeds.dtype,
			device,
			generator,
			latents,
		)
		edit_momentum = None

		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):
				# latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
				latent_model_input = latents
				latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

				noise_pred = self._ours_step(latent_model_input, prompt_embeds, t, i)

				noise_pred_uncond, noise_pred_text = noise_pred[0][None], noise_pred[-1][None]
				noise_pred_concepts = noise_pred[1:-1]

				prompt_score = (noise_pred_text-noise_pred_uncond)
				subject_scores = (noise_pred_concepts - noise_pred_uncond)
				subject_scores = extract_concept(subject_scores, subject_percentile)
				# abstract_score = get_perpendicular_component(extract_concept(prompt_score, 0.5)[0], torch.sum(subject_scores, dim=0))[None]
				# abstract_score = get_perpendicular_component(prompt_score[0], torch.sum(subject_scores, dim=0))[None]
				abstract_score = prompt_score[0]
				for s in subject_scores:
					abstract_score = get_perpendicular_component(abstract_score, s)
				abstract_score = abstract_score.unsqueeze(0)
				concept_scores = torch.cat([subject_scores, abstract_score], dim=0)

				sim = torch.abs(torch.nn.functional.cosine_similarity(
					prompt_score.reshape(1, -1), concept_scores.reshape(len(prompt)-1, -1)
				)).detach().cpu().numpy()
				self.sim.append(sim)

				if edit_momentum is None:
					edit_momentum = torch.zeros_like(concept_scores)

				concept_guidance = 0
				concept_guidance -= (prompt_score - abstract_score)[0] / concept_scores.size(0)
				start = 0 if i < window else i - window 
				for idx, concept_score in enumerate(concept_scores):
					if idx != concept_scores.size(0):
						_concept_guidance = get_perpendicular_component(concept_score, prompt_score[0]) + edit_momentum_scale * edit_momentum[idx]
						# _concept_guidance = concept_score + edit_momentum_scale * edit_momentum[idx]
						edit_momentum[idx] = edit_mom_beta * edit_momentum[idx] + (1 - edit_mom_beta) * _concept_guidance

						if i > warmup and i < cooldown and np.mean(np.array(self.sim).transpose()[idx, start:i]) < subject_th:
							concept_guidance += _concept_guidance 

					else:
						_concept_guidance = abstract_score + edit_momentum_scale * edit_momentum[idx]
						edit_momentum[idx] = edit_mom_beta * edit_momentum[idx] + (1 - edit_mom_beta) * _concept_guidance

						if i > warmup and i < cooldown and np.mean(np.array(self.sim).transpose()[idx, start:i]) < abstract_th:
							concept_guidance += _concept_guidance 

				noise_pred = noise_pred_uncond + guidance_scale * prompt_score \
							+ concept_guidance_scale * concept_guidance 

				latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
				
				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						callback(i, t, latents)

		if output_type == "latent":
			image = latents
			has_nsfw_concept = None
		elif output_type == "pil":
			# 8. Post-processing
			image = self.decode_latents(latents)

			# 9. Run safety checker
			# image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

			# 10. Convert to PIL
			image = self.numpy_to_pil(image)
		else:
			# 8. Post-processing
			image = self.decode_latents(latents)

			# 9. Run safety checker
			# image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

		# Offload last model to CPU
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.final_offload_hook.offload()

		if not return_dict:
			return (image, has_nsfw_concept)

		return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


if __name__ == "__main__":
	import yaml
	import os
	from PIL import Image
	from diffusers import DDIMScheduler, LMSDiscreteScheduler

	from utils.util import *
	from utils.ptp_util import AttentionStore
	from utils.vis_util import visualize_cross_attention_map
	from utils.data_util import load_data

	from pipelines.stable_diffusion_pipeline import StableDiffusionPipeline

	config = yaml.safe_load(open("configs/models/ours.yaml",'r'))
	# config["params"]["num_inference_steps"] = 25
	
	scheduler = DDIMScheduler.from_pretrained(config["version"], subfolder="scheduler")
	model = OurPipeline.from_pretrained(config["version"]).to("cuda")
	# model.scheduler = scheduler

	attention_store = AttentionStore()
	register_attention_control(model, attention_store, None)

	data = load_data("data/CC-500.csv", "ours")

	os.makedirs("tmp", exist_ok=True)
	c = 0
	for p in data[50:]:
		prompt, nps = p
		seed = np.random.randint(0,1000)
		g = torch.Generator("cuda").manual_seed(seed)
		img = model(prompt, nps, generator=g, **config["params"]).images[0]
		img.save('tmp/tmp.png')
		model.plot_similarities("tmp/tmp2.png")
		del model

		model = StableDiffusionPipeline.from_pretrained(config["version"]).to("cuda")
		g = torch.Generator("cuda").manual_seed(seed)
		img = model(prompt=prompt, generator=g).images[0]
		img.save('tmp/tmp0.png')
		img0 = np.array(Image.open('tmp/tmp0.png'))

		img = np.array(Image.open('tmp/tmp.png'))#.transpose(2,1,0)
		graph = Image.open('tmp/tmp2.png')
		graph = np.array(graph.resize((512,512)))[:,:,:3]#.transpose(2,1,0)[:3,:,:]
		# print(img.shape, graph.shape)
		Image.fromarray(np.hstack([img0,img, graph])).save(f'tmp/{seed}_{prompt}.png')
		os.remove('tmp/tmp0.png')
		os.remove('tmp/tmp.png')
		os.remove('tmp/tmp2.png')
		c+=1
		if c == 5:
			break

	# prompt = "a brown bench and a green clock"
	# noun_phrases = ["a brown bench", "a green clock"]
	# # prompt = "a pink banana and a green cake"
	# # noun_phrases = ["a pink banana", "a green cake"]

	# seed = np.random.randint(0, 1000)
	# seed = 153# 779 #464
	# g = torch.Generator("cuda").manual_seed(seed)

	# config["params"]["num_inference_steps"] = 25
	# img = model(prompt, noun_phrases, generator=g, **config["params"]).images[0]
	# img.save('tmp.png')
	# model.plot_similarities("tmp3.png")
	# visualize_cross_attention_map(attention_store, 
	# 	img, 
	# 	model.tokens, 
	# 	**config["vis_option"],
	# 	out="tmp2.png")