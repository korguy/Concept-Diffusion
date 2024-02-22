from .common import CONFIGS_DIR, extract_nouns, nlp_for_extract_nouns
from ..pipelines import model_dict
from ..utils.ptp_util import AttentionStore, register_attention_control

import torch
from torchvision.transforms import ToTensor
import numpy as np
import spacy
from PIL import Image

import os
import os.path as osp
from pathlib import Path
from abc import ABC, abstractmethod
import random
import json
import typing


SCHEDULERS = ("ddim", "pndm", "ddpm")

PIPELINE_TYPENAME = "DIFFUSERS_PIPELINE"
PARAMSDICT_TYPENAME = "SAMPLER_PARAMS"
# Attend and Excite
ATTENTIONSTORE_TYPENAME = "ATTENTION_STORE"

CATEGORY_BASE = "Concept Guidance"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# This must match 'model' in model config yaml
MODEL_NAME_STABLE_DIFFUSION = "stable_diffusion"
MODEL_NAME_COMPOSABLE_DIFFUSION = "composable_diffusion"
MODEL_NAME_ATTEND_AND_EXCITE = "attend_and_excite"

MODEL_DISPLAY_NAME_STABLE_DIFFUSION = "Stable Diffusion"
MODEL_DISPLAY_NAME_COMPOSABLE_DIFFUSION = "Composable Diffusion"
MODEL_DISPLAY_NAME_ATTEND_AND_EXCITE = "Attend and Excite"


class BuildPipelineBase(ABC):
    @classmethod
    def _get_required_input_types(clazz):
        if not clazz.MODEL_NAME:
            raise ValueError("MODEL_NAME is not set")
        return {
            "config": (os.listdir(CONFIGS_DIR), { 
                "concept_guidance_model_config": True,
                "model_name": clazz.MODEL_NAME, # 'model' in config yaml
            }),
            "name": ("STRING", {"default": ""}),
            "version": ("STRING", {"default": ""}),
            "scheduler": (SCHEDULERS, ),
        }

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz._get_required_input_types(),
        }

    MODEL_NAME = None  # Concrete subclasses MUST override this field
    MODEL_DISPLAY_NAME = None

    CATEGORY = CATEGORY_BASE
    FUNCTION = "build_pipeline_fun"
    RETURN_TYPES = (PIPELINE_TYPENAME, )
    RETURN_NAMES = ("pipeline", )

    def build_pipeline_fun(self, config, name, version, scheduler, **opts):
        return (self.get_pipeline(config, name, version, scheduler, **opts), )

    def get_pipeline(self, config, name, version, scheduler, **opts):
        return model_dict[self.MODEL_NAME].from_pretrained(version).to(device)

    @classmethod
    def IS_CHANGED(clazz, config, name, version, scheduler, **opts):
        return f"{config}_{name}_{version}_{scheduler}_{opts}"

    @classmethod
    def VALIDATE_INPUTS(clazz, config, name, version, scheduler, **opts):
        if not Path(CONFIGS_DIR, config).is_file():
            return "Invalid config: '{}'".format(config)
        return True

    @classmethod
    def get_node_display_name(clazz):
        return "Build Pipeline ({})".format(
            clazz.MODEL_DISPLAY_NAME if clazz.MODEL_DISPLAY_NAME is not None else clazz.MODEL_NAME)


class LoadParamsBase(ABC):
    @classmethod
    def _get_required_input_types(clazz):
        if not clazz.MODEL_NAME:
            raise ValueError("MODEL_NAME is not set")
        return {
            "config": (os.listdir(CONFIGS_DIR), { 
                "concept_guidance_model_config": True,
                "model_name": clazz.MODEL_NAME, # 'model' in config yaml
            }),

            "param_width": ("INT", {"default": 512, "min": 1, "max": 2048}),
            "param_height": ("INT", {"default": 512, "min": 1, "max": 2048}),
            "param_guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 100}),
            "param_num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
        }

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz._get_required_input_types(),
        }

    MODEL_NAME = None  # Concrete subclasses MUST override this field
    MODEL_DISPLAY_NAME = None

    CATEGORY = CATEGORY_BASE
    FUNCTION = "load_params_fun"
    RETURN_TYPES = (PARAMSDICT_TYPENAME, )
    RETURN_NAMES = ("sampler_params", )

    def load_params_fun(self, config, **opts):
        return ({
            "model_name": self.MODEL_NAME,
            **self._get_params(opts)}, )

    @classmethod
    def IS_CHANGED(clazz, config, **opts):
        return f"{config}_{opts}"

    @classmethod
    def VALIDATE_INPUTS(clazz, config, **opts):
        if not Path(CONFIGS_DIR, config).is_file():
            return "Invalid config: '{}'".format(config)
        return True

    @classmethod
    def _get_params(clazz, d: dict):
        return {k[len("param_"):]: v for k, v in d.items() if k.startswith("param_")}

    @classmethod
    def get_node_display_name(clazz):
        return "Load Params ({})".format(
            clazz.MODEL_DISPLAY_NAME if clazz.MODEL_DISPLAY_NAME is not None else clazz.MODEL_NAME)


class GenerateImageBase(ABC):
    REQUIRED_INPUT_TYPES = {
        "pipeline": (PIPELINE_TYPENAME, {}),
        "sampler_params": (PARAMSDICT_TYPENAME, {}),
        "prompt": ("STRING", {"multiline": True}),
        "seed": ("INT", {"default": -1}),
    }

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz.REQUIRED_INPUT_TYPES,
        }

    MODEL_NAME = None  # Concrete subclasses MUST override this field
    MODEL_DISPLAY_NAME = None

    CATEGORY = CATEGORY_BASE
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )

    def generate(self, pipeline, sampler_params, prompt, seed, **opts):
        if not self.MODEL_NAME or self.MODEL_NAME != sampler_params.get('model_name', None):
            raise ValueError("model name mismatch ('{}':'{}')".format(
                self.MODEL_NAME, sampler_params.get('model_name', None)))
        images = self.get_images(
                pipeline, sampler_params=sampler_params, prompt=prompt, seed=seed, **opts)
        return (self._images_to_tensors(images), )

    def get_images(self, pipeline, sampler_params, prompt, seed, **opts) -> typing.List[Image.Image]:
        return pipeline(
                   prompt=prompt,
                   generator=self._g(seed), **sampler_params).images


    @classmethod
    def _images_to_tensors(clazz, images):
        return torch.stack([torch.permute(ToTensor()(image), (1, 2, 0)) for image in images])

    @classmethod
    def _g(clazz, seed):
        return torch.Generator(device).manual_seed(seed if seed >= 0 else random.randint())

    @classmethod
    def get_node_display_name(clazz):
        return "Generate ({})".format(
            clazz.MODEL_DISPLAY_NAME if clazz.MODEL_DISPLAY_NAME is not None else clazz.MODEL_NAME)


class BuildPipeline_StableDiffusion(BuildPipelineBase):
    MODEL_NAME = MODEL_NAME_STABLE_DIFFUSION
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_STABLE_DIFFUSION


class LoadParams_StableDiffusion(LoadParamsBase):
    MODEL_NAME = MODEL_NAME_STABLE_DIFFUSION
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_STABLE_DIFFUSION

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz._get_required_input_types(),
            "optional": {
                "param_eta": ("FLOAT", {"default": 0.0}),
                "param_guidance_rescale": ("FLOAT", {"default": 0.0}),
            }
        }


class GenerateImage_StableDiffusion(GenerateImageBase):
    MODEL_NAME = MODEL_NAME_STABLE_DIFFUSION
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_STABLE_DIFFUSION


class BuildPipeline_ComposableDiffusion(BuildPipelineBase):
    MODEL_NAME = MODEL_NAME_COMPOSABLE_DIFFUSION
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_COMPOSABLE_DIFFUSION


class LoadParams_ComposableDiffusion(LoadParamsBase):
    MODEL_NAME = MODEL_NAME_COMPOSABLE_DIFFUSION
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_COMPOSABLE_DIFFUSION

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz._get_required_input_types(),
            "optional": {
                "param_eta": ("FLOAT", {"default": 0.0}),
                "param_guidance_rescale": ("FLOAT", {"default": 0.0}),
            }
        }


class PreprocessPromptForComposable:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": {
                "raw_prompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("composable_text", )

    FUNCTION = "preprocess"
    def preprocess(self, raw_prompt):
        if not raw_prompt:
            return ("", )
        else:
            doc = self.nlp(raw_prompt)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            print("OUT", "|".join(noun_phrases))
            return ("|".join(noun_phrases), )

    @classmethod
    def get_node_display_name(clazz):
        return "Preprocess prompt for Composable Diffusion"


class GenerateImage_ComposableDiffusion(GenerateImageBase):
    MODEL_NAME = MODEL_NAME_COMPOSABLE_DIFFUSION
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_COMPOSABLE_DIFFUSION


class BuildPipeline_AttendAndExcite(BuildPipelineBase):
    MODEL_NAME = MODEL_NAME_ATTEND_AND_EXCITE
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_ATTEND_AND_EXCITE

    RETURN_TYPES = (PIPELINE_TYPENAME, ATTENTIONSTORE_TYPENAME)
    RETURN_NAMES = ("pipeline", "attention_store")

    def build_pipeline_fun(self, config, name, version, scheduler, **opts):
        pipe = model_dict[self.MODEL_NAME].from_pretrained(version).to(device)
        attention_store = AttentionStore() 
        register_attention_control(pipe, attention_store)

        return (pipe, attention_store)


class LoadParams_AttendAndExcite(LoadParamsBase):
    MODEL_NAME = MODEL_NAME_ATTEND_AND_EXCITE
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_ATTEND_AND_EXCITE

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": {
                **clazz._get_required_input_types(),
                "param_attention_res": ("INT", {"default": 16}),
                "param_max_iter_to_alter": ("INT", {"default": 25}),
                "param_sigma": ("FLOAT", {"default": 0.5}),
                "param_kernel_size": ("INT", {"default": 3}),
                "param_smooth_attentions": ("BOOLEAN", {"default": True}),
                "param_thresholds": ("STRING", {"default": json.dumps({0: 0.05, 10: 0.5, 20: 0.8}), "multiline": True}),
                "param_scale_factor": ("INT", {"default": 20}),
            },
            "optional": {
                "param_eta": ("FLOAT", {"default": 0.0}),
                "param_guidance_rescale": ("FLOAT", {"default": 0.0}),
            }
        }


class ExtractNounsForAttendAndExcite:
    def __init__(self):
        self.nlp = None

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "do_extract_nouns_on_exec": ("BOOLEAN", {"default": True}),
                "nouns_list": ("STRING", {
                        "extracted_nouns_json": True,
                        "multiline": True,
                    }),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("nouns_list", )

    FUNCTION = "extract_nouns_fun"
    def extract_nouns_fun(self, prompt, do_extract_nouns_on_exec: bool, nouns_list: str):
        nouns = None # List[str]
        if do_extract_nouns_on_exec:
            if self.nlp is None:
                self.nlp = nlp_for_extract_nouns
            nouns = extract_nouns(prompt or "", nlp=self.nlp)
        else:
            nouns = json.loads(nouns_list)

        nouns = list(map(str, nouns))
        return (json.dumps(nouns), )

    @classmethod
    def VALIDATE_INPUTS(clazz, prompt, do_extract_nouns_on_exec: bool, nouns_list: str):
        if not do_extract_nouns_on_exec:
            try:
                json.loads(nouns_list)
            except JSONDecodeError:
                return "Given nouns_list '{}' is not a proper JSON list.".format(nouns_list)
        return True

    @classmethod
    def get_node_display_name(clazz):
        return "Extract nouns"


class GenerateImage_AttendAndExcite(GenerateImageBase):
    MODEL_NAME = MODEL_NAME_ATTEND_AND_EXCITE
    MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAME_ATTEND_AND_EXCITE

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": {
                **clazz.REQUIRED_INPUT_TYPES,
                "attention_store": (ATTENTIONSTORE_TYPENAME, {"forceInput": True}),
                "nouns": ("STRING", {"default": json.dumps([])}),
            },
        }

    def get_images(self, pipeline, sampler_params, prompt, seed, **opts) -> typing.List[Image.Image]:
        nouns = json.loads(opts['nouns'])
        attention_store = opts['attention_store']
        print("{} - nouns (#={}): {}".format(self.__class__.__name__, len(nouns), nouns))
        token_indices = pipeline.get_indices(prompt)
        indices_to_alter = []
        for n in nouns:
            try:
                indices_to_alter.append(token_indices[n])
            except:
                continue

        if 'thresholds' in sampler_params:
            th_str = sampler_params['thresholds']
            if not th_str.startswith("{"):  # Curly braces are stripped for some reason?
                th_str = "{" + th_str + "}"
            sampler_params['thresholds'] = json.loads(th_str)

        return pipeline(
                prompt=prompt,
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                generator=self._g(seed),
                **{**sampler_params,
                   'return_dict': True,
                   'output_type': 'pil'}
            ).images


node_classes = [
        BuildPipeline_StableDiffusion,
        BuildPipeline_ComposableDiffusion,
        BuildPipeline_AttendAndExcite,

        LoadParams_StableDiffusion,
        LoadParams_ComposableDiffusion,
        LoadParams_AttendAndExcite,

        GenerateImage_StableDiffusion,
        GenerateImage_ComposableDiffusion,
        GenerateImage_AttendAndExcite,

        PreprocessPromptForComposable,
        ExtractNounsForAttendAndExcite,
]

get_node_identifier = lambda clazz: clazz.__name__
get_node_display_name = lambda clazz: getattr(clazz, "get_node_display_name")() \
        if hasattr(clazz, "get_node_display_name") else get_node_identifier(clazz)

NODE_CLASS_MAPPINGS = {get_node_identifier(clazz): clazz for clazz in node_classes}
NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    # ...
}

NODE_DISPLAY_NAME_MAPPINGS = \
        {get_node_identifier(clazz): get_node_display_name(clazz) for clazz in node_classes}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    # ...
}
