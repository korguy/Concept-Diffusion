from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


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

	@torch.no_grad()
	def __call__(
		self,
		prompt: str,
		noun_phrases: List[str], 
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
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
		)

		self.tokens = tokens

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

		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):
				latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
				latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

				noise_pred = self.unet(
					latent_model_input,
					t,
					encoder_hidden_states=prompt_embeds,
					cross_attention_kwargs=cross_attention_kwargs,
					return_dict=False
				)[0]

				if do_classifier_free_guidance:
					noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
					noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

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
			image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

			# 10. Convert to PIL
			image = self.numpy_to_pil(image)
		else:
			# 8. Post-processing
			image = self.decode_latents(latents)

			# 9. Run safety checker
			image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

		# Offload last model to CPU
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.final_offload_hook.offload()

		if not return_dict:
			return (image, has_nsfw_concept)

		return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


if __name__ == "__main__":
	import yaml
	from diffusers import DDIMScheduler

	from utils.util import *
	from utils.ptp_util import AttentionStore

	config = yaml.safe_load(open("configs/models/ours.yaml",'r'))

	scheduler = DDIMScheduler.from_pretrained(config["version"], subfolder="scheduler")
	model = OurPipeline.from_pretrained(config["version"]).to("cuda")

	attention_store = AttentionStore()
	register_attention_control(model, attention_store, None)

	prompt = "a green backpack and a brown suitcase"
	noun_phrases = ["a green backpack", "a brown suitcase"]

	img = model(prompt, noun_phrases, **config["params"]).images[0]
	img.save('tmp.png')