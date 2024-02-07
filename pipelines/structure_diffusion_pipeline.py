import inspect
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Dict

import numpy as np
import stanza
from nltk.tree import Tree
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers.tokenization_utils import BatchEncoding
from utils.structure_util import *

@dataclass
class Span(object):
    left: int
    right: int


@dataclass
class SubNP(object):
    text: str
    span: Span


@dataclass
class AllNPs(object):
    nps: List[str]
    spans: List[Span]
    lowest_nps: List[SubNP]

class StructureDiffusionPipeline(StableDiffusionPipeline):

    def preprocess_prompt(self, prompt: str) -> str:
        return prompt.lower().strip().strip(".").strip()

    def get_sub_nps(
        self,
        tree: Tree,
        full_sent: str,
        left: int,
        right: int,
        idx_map: Dict[int, List[int]],
        highest_only: bool = False,
    ) -> List[SubNP]:

        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []

        sub_nps: List[SubNP] = []

        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[: len(n_subtree_leaves)]
        assert right - left == n_leaves

        if tree.label() == "NP" and n_leaves > 1:
            sub_np = SubNP(
                text=" ".join(tree.leaves()),
                span=Span(left=int(min(idx_map[left])), right=int(min(idx_map[right]))),
            )
            sub_nps.append(sub_np)

            if highest_only and sub_nps[-1].text != full_sent:
                return sub_nps

        for i, subtree in enumerate(tree):
            sub_nps += self.get_sub_nps(
                subtree,
                full_sent,
                left=left + offset[i],
                right=left + offset[i] + n_subtree_leaves[i],
                idx_map=idx_map,
            )
        return sub_nps

    def get_token_alignment_map(
        self, tree: Tree, tokens: Optional[List[str]]
    ) -> Dict[int, List[int]]:
        if tokens is None:
            return {i: [i] for i in range(len(tree.leaves()) + 1)}

        def _get_token(token: str):
            return token[:-4] if token.endswith("</w>") else token

        idx_map: Dict[int, List[int]] = {}
        j = 0
        max_offset = abs(len(tokens) - len(tree.leaves()))
        tree_prev_leaf = ""
        for i, w in enumerate(tree.leaves()):
            token = _get_token(tokens[j])
            idx_map[i] = [j]
            if token == tree_prev_leaf + w:
                tree_prev_leaf = ""
                j += 1
            else:
                if len(token) < len(w):
                    prev = ""
                    while prev + token != w:
                        prev += token
                        j += 1
                        token = _get_token(tokens[j])
                        idx_map[i].append(j)
                        assert j - i <= max_offset
                else:
                    tree_prev_leaf += w
                    j -= 1
                j += 1
        idx_map[i + 1] = [j]
        return idx_map

    def get_all_nps(
        self,
        tree: Tree,
        full_sent: str,
        tokens: Optional[List[str]] = None,
        highest_only: bool = False,
        lowest_only: bool = False,
    ) -> AllNPs:
        start = 0
        end = len(tree.leaves())

        idx_map = self.get_token_alignment_map(tree=tree, tokens=tokens)

        all_sub_nps = self.get_sub_nps(
            tree,
            full_sent,
            left=start,
            right=end,
            idx_map=idx_map,
            highest_only=highest_only,
        )

        lowest_nps: List[SubNP] = []
        for i in range(len(all_sub_nps)):
            span = all_sub_nps[i].span
            lowest = True
            for j in range(len(all_sub_nps)):
                span2 = all_sub_nps[j].span
                if span2.left >= span.left and span2.right <= span.right:
                    lowest = False
                    break
            if lowest:
                lowest_nps.append(all_sub_nps[i])
        
        if lowest_only:
            all_nps = [lowest_np.text for lowest_np in lowest_nps]

        all_nps = [all_sub_np.text for all_sub_np in all_sub_nps]
        spans = [all_sub_np.span for all_sub_np in all_sub_nps]

        if full_sent and full_sent not in all_nps:
            all_nps = [full_sent] + all_nps
            spans = [Span(left=start, right=end)] + spans

        return AllNPs(nps=all_nps, spans=spans, lowest_nps=lowest_nps)

    def tokenize(self, prompt: Union[str, List[str]]) -> BatchEncoding:
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_input

    def _extend_string(self, nps: List[str]) -> List[str]:
        extend_nps: List[str] = []
        for i in range(len(nps)):
            if i == 0:
                extend_nps.append(nps[i])
            else:
                np = (" " + nps[i]) * (
                    self.tokenizer.model_max_length // len(nps[i].split())
                )
                extend_nps.append(np)
        return extend_nps

    def _expand_sequence(
        self, seq: torch.Tensor, length: int, dim: int = 1
    ) -> torch.Tensor:

        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        max_length = seq.size(0)
        n_repeat = (max_length - 2) // length

        # shape: (10, 1)
        repeat_size = (n_repeat,) + (1,) * (len(seq.size()) - 1)

        # shape: (77,)
        eos = seq[length + 1, ...].clone()

        # shape: (750, 77)
        segment = seq[1 : length + 1, ...].repeat(*repeat_size)

        seq[1 : len(segment) + 1] = segment

        # To avoid the following error, we need to use `torch.no_grad` function:
        # RuntimeError: Output 0 of SliceBackward0 is a view and
        # # is being modified inplace. This view is the output
        # of a function that returns multiple views.
        # Such functions do not allow the output views to be modified inplace.
        # You should replace the inplace operation by an out-of-place one.
        seq[len(segment) + 1] = eos

        # shape: (768, 77) -> (77, 768)
        return seq.transpose(0, dim)

    def _align_sequence(
        self,
        full_seq: torch.Tensor,
        seq: torch.Tensor,
        span: Span,
        eos_loc: int,
        dim: int = 1,
        zero_out: bool = False,
        replace_pad: bool = False,
    ) -> torch.Tensor:

        # shape: (77, 768) -> (768, 77)
        seq = seq.transpose(0, dim)

        # shape: (77, 768) -> (768, 77)
        full_seq = full_seq.transpose(0, dim)

        start, end = span.left + 1, span.right + 1
        seg_length = end - start

        full_seq[:, start:end] = seq[:, 1 : 1 + seg_length]
        if zero_out:
            full_seq[:, 1:start] = 0
            full_seq[:, end:eos_loc] = 0

        if replace_pad:
            pad_length = len(full_seq) - eos_loc
            full_seq[:, eos_loc:] = seq[:, 1 + seg_length : 1 + seg_length + pad_length]

        # shape: (768, 77) -> (77, 768)
        return full_seq.transpose(0, dim)

    def extend_str(self, nps: List[str]) -> torch.Tensor:
        nps = self._extend_string(nps)

        input_ids = self.tokenize(nps).input_ids
        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state
        return c

    def extend_seq(self, nps: List[str]):

        input_ids = self.tokenize(nps).input_ids

        # repeat each NP after embedding
        nps_length = [len(ids) - 2 for ids in input_ids]  # not including bos eos

        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        c = torch.stack(
            [c[0]]
            + [self._expand_sequence(seq, l) for seq, l in zip(c[1:], nps_length[1:])]
        )
        return c

    def align_seq(self, nps: List[str], spans: List[Span]) -> KeyValueTensors:

        input_ids = self.tokenize(nps).input_ids
        nps_length = [len(ids) - 2 for ids in input_ids]
        enc_output = self.text_encoder(input_ids.to(self.device))
        c = enc_output.last_hidden_state

        # shape: (num_nps, model_max_length, hidden_dim)
        k_c = torch.stack(
            [c[0]]
            + [
                self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
                for seq, span in zip(c[1:], spans[1:])
            ]
        )
        # shape: (num_nps, model_max_length, hidden_dim)
        v_c = torch.stack(
            [c[0]]
            + [
                self._align_sequence(c[0].clone(), seq, span, nps_length[0] + 1)
                for seq, span in zip(c[1:], spans[1:])
            ]
        )
        return KeyValueTensors(k=k_c, v=v_c)

    def apply_text_encoder(
        self,
        struct_attention: STRUCT_ATTENTION_TYPE,
        prompt: str,
        nps: List[str],
        spans: Optional[List[Span]] = None,
    ) -> Union[torch.Tensor, KeyValueTensors]:

        if struct_attention == "extend_str":
            return self.extend_str(nps=nps)

        elif struct_attention == "extend_seq":
            return self.extend_seq(nps=nps)

        elif struct_attention == "align_seq" and spans is not None:
            return self.align_seq(nps=nps, spans=spans)

        elif struct_attention == "none":
            text_input = self.tokenize(prompt)
            return self.text_encoder(text_input.input_ids.to(self.device))[0]

        else:
            raise ValueError(f"Invalid type of struct attention: {struct_attention}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        struct_attention: STRUCT_ATTENTION_TYPE = "none",
        **kwargs,
    ) -> StableDiffusionPipelineOutput:

        device = self.device

        replace_cross_attention(nn_module=self.unet, name="unet")

        self.nlp = stanza.Pipeline(
            lang="en", processors="tokenize,pos,constituency", use_gpu=False
        )
        batch_size = 1
        preprocessed_prompt = self.preprocess_prompt(prompt)

        doc = self.nlp(preprocessed_prompt)
        tree = Tree.fromstring(str(doc.sentences[0].constituency))
        tokens = self.tokenizer.tokenize(preprocessed_prompt)
        all_nps = self.get_all_nps(
            tree=tree, full_sent=preprocessed_prompt, tokens=tokens
        )
        cond_embeddings = self.apply_text_encoder(
            struct_attention=struct_attention,
            prompt=preprocessed_prompt,
            nps=all_nps.nps,
            spans=all_nps.spans,
        )

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            uncond_input = self.tokenize([""] * batch_size)
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            if struct_attention == "align_seq":
                text_embeddings = (uncond_embeddings, cond_embeddings)
            else:
                text_embeddings = (uncond_embeddings, cond_embeddings)
        
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            uncond_embeddings.dtype,
            device,
            generator,
            None,
        )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            # compute the previous noisy sample x_t -> x_t-1
            # if isinstance(self.scheduler, LMSDiscreteScheduler):
            #     latents = self.scheduler.step(
            #         noise_pred, i, latents, **extra_step_kwargs
            #     ).prev_sample
            # else:
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample
        
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, uncond_embeddings.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, uncond_embeddings.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)



if __name__ == "__main__":
    model = StructureDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model = model.to("cuda")

    prompt = "a red car and a white sheep"
    image = model(prompt, struct_attention="align_seq").images[0]
    image.save('tmp.png')