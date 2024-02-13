from typing import Union, Tuple, List

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from diffusers.models.cross_attention import CrossAttention

class SimilarityStore:
	def __init__(self, steps, counts):
		super().__init__()

class CustomAttnProcessor:
	def __init__(self, attnstore, sim_store, place_in_unet):
		super().__init__()
		self.attnstore = attnstore
		self.sim_store = sim_store
		self.place_in_unet = place_in_unet

	def self_attention(self, attn, hidden_states, attention_mask):
		query = attn.to_q(hidden_states)
		key = attn.to_k(hidden_states)
		value = attn.to_v(hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)
		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)

		return hidden_states

	def cross_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask):
		encoder_hidden_states = torch.cat([encoder_hidden_states[0].unsqueeze(0),
												encoder_hidden_states[-1].unsqueeze(0)
												], dim=0)

		query = attn.to_q(hidden_states)
		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		query = attn.head_to_batch_dim(query)
		key = attn.head_to_batch_dim(key)
		value = attn.head_to_batch_dim(value)

		attention_probs = attn.get_attention_scores(query, key, attention_mask)
		hidden_states = torch.bmm(attention_probs, value)
		hidden_states = attn.batch_to_head_dim(hidden_states)

		return hidden_states

	def __call__(
		self,
		attn: CrossAttention,
		hidden_states, # diffusion latents z
		encoder_hidden_states=None, # CLIP text embeddings
		attention_mask=None
	):
		batch_size, sequence_length, _ = hidden_states.shape
		attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

		is_cross = encoder_hidden_states is not None
		if not is_cross:
			hidden_states = self.self_attention(attn, hidden_states, attention_mask)
		else:
			hidden_states = self.cross_attention(attn, hidden_states, encoder_hidden_states, attention_mask)


		query = attn.to_q(hidden_states)

		is_cross = encoder_hidden_states is not None
		if is_cross:
			encoder_hidden_states = torch.cat([encoder_hidden_states[0].unsqueeze(0),
												encoder_hidden_states[-1].unsqueeze(0)
												], dim=0)
		else:
			encoder_hidden_states = hidden_states

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		return hidden_states

def register_attention_control(model, controller, similarity):
	attn_procs = {}
	cross_att_count = 0

	for name in model.unet.attn_processors.keys():
		cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
		if name.startswith("mid_block"):
			hidden_size = model.unet.config.block_out_channels[-1]
			place_in_unet = "mid"
		elif name.startswith("up_blocks"):
			block_id = int(name[len("up_blocks.")])
			hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
			place_in_unet = "up"
		elif name.startswith("down_blocks"):
			block_id = int(name[len("down_blocks.")])
			hidden_size = model.unet.config.block_out_channels[block_id]
			place_in_unet = "down"
		else:
			continue

		cross_att_count += 1
		attn_procs[name] = CustomAttnProcessor(
			attnstore=controller, sim_store=similarity, place_in_unet=place_in_unet
		)

	model.unet.set_attn_processor(attn_procs)
	controller.num_att_layers = cross_att_count