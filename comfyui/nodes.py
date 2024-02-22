from .common import CONFIGS_DIR
from ..pipelines import model_dict

import torch
from torchvision.transforms import ToTensor
import numpy as np
import spacy

import os
import os.path as osp
from pathlib import Path
from abc import ABC, abstractmethod
import random


SCHEDULERS = ("ddim", "pndm", "ddpm")

PIPELINE_TYPENAME = "DIFFUSERS_PIPELINE"
PARAMSDICT_TYPENAME = "SAMPLER_PARAMS"

CATEGORY_BASE = "Concept Guidance"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

MODEL_NAME_STABLE_DIFFUSION = "stable_diffusion"
MODEL_NAME_COMPOSABLE_DIFFUSION = "composable_diffusion"


class SetUpPipelineBase(ABC):

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

    CATEGORY = CATEGORY_BASE
    FUNCTION = "setup"
    RETURN_TYPES = (PIPELINE_TYPENAME, PARAMSDICT_TYPENAME)
    RETURN_NAMES = ("pipeline", "sampler_params")

    def setup(self, config, name, version, scheduler,
            param_width, param_height, param_guidance_scale, param_num_inference_steps, **opts):
        return (self.get_pipeline(config, name, version, scheduler, **opts),
                {"model_name": self.MODEL_NAME,
                    **self._get_params({
                        "param_guidance_scale": param_guidance_scale,
                        "param_num_inference_steps": param_num_inference_steps,
                        **opts})})

    @abstractmethod
    def get_pipeline(self, config, name, version, scheduler, **opts):
        pass

    @classmethod
    def IS_CHANGED(clazz, config, name, version, scheduler,
            param_width, param_height, param_guidance_scale, param_num_inference_steps, **opts):
        return f"{config}_{name}_{version}_{scheduler}_{param_width}_{param_height}" + \
               f"_{param_guidance_scale}_{param_num_inference_steps}_{opts}"

    @classmethod
    def VALIDATE_INPUTS(clazz, config, name, version, scheduler,
            param_width, param_height, param_guidance_scale, param_num_inference_steps, **opts):
        if not Path(CONFIGS_DIR, config).is_file():
            return "Invalid config: '{}'".format(config)
        return True

    @classmethod
    def _get_params(clazz, d: dict):
        return {k[len("param_"):]: v for k, v in d.items() if k.startswith("param_")}


class SetUpPipeline_StableDiffusion(SetUpPipelineBase):
    MODEL_NAME = MODEL_NAME_STABLE_DIFFUSION

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz._get_required_input_types(),
            "optional": {
                "param_eta": ("FLOAT", {"default": 0.0}),
                "param_guidance_rescale": ("FLOAT", {"default": 0.0}),
            }
        }

    def get_pipeline(self, config, name, version, scheduler, **opts):
        return model_dict[self.MODEL_NAME].from_pretrained(version).to(device)


class SetUpPipeline_ComposableDiffusion(SetUpPipelineBase):
    MODEL_NAME = MODEL_NAME_COMPOSABLE_DIFFUSION

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz._get_required_input_types(),
            "optional": {
                "param_eta": ("FLOAT", {"default": 0.0}),
                "param_guidance_rescale": ("FLOAT", {"default": 0.0}),
            }
        }

    def get_pipeline(self, config, name, version, scheduler, **opts):
        return model_dict[self.MODEL_NAME].from_pretrained(version).to(device)


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
        doc = self.nlp(raw_prompt)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        print("OUT", "|".join(noun_phrases))
        return ("|".join(noun_phrases), )


class GenerateImageBase(ABC):
    REQUIRED_INPUT_TYPES = {
        "pipeline": (PIPELINE_TYPENAME, {}),
        "sampler_params": (PARAMSDICT_TYPENAME, {}),
        "prompt": ("STRING", {"default": ""}),
        "seed": ("INT", {"default": -1}),
    }

    @classmethod
    def INPUT_TYPES(clazz):
        return {
            "required": clazz.REQUIRED_INPUT_TYPES,
        }

    MODEL_NAME = None  # Concrete subclasses MUST override this field

    CATEGORY = CATEGORY_BASE
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )

    def generate(self, pipeline, sampler_params, prompt, seed, **opts):
        if not self.MODEL_NAME or self.MODEL_NAME != sampler_params.get('model_name', None):
            raise ValueError("model name mismatch ('{}':'{}')".format(
                self.MODEL_NAME, sampler_params.get('model_name', None)))
        return (self._images_to_tensors(pipeline(
                   prompt=prompt,
                   generator=self._g(seed), **sampler_params).images), )

    @classmethod
    def _images_to_tensors(clazz, images):
        return torch.stack([torch.permute(ToTensor()(image), (1, 2, 0)) for image in images])

    @classmethod
    def _g(clazz, seed):
        return torch.Generator(device).manual_seed(seed if seed >= 0 else random.randint())



class GenerateImage_StableDiffusion(GenerateImageBase):
    MODEL_NAME = MODEL_NAME_STABLE_DIFFUSION


class GenerateImage_ComposableDiffusion(GenerateImageBase):
    MODEL_NAME = MODEL_NAME_COMPOSABLE_DIFFUSION


NODE_CLASS_MAPPINGS = {
    "SetUpPipeline_StableDiffusion": SetUpPipeline_StableDiffusion,
    "SetUpPipeline_ComposableDiffusion": SetUpPipeline_ComposableDiffusion,

    "GenerateImage_StableDiffusion": GenerateImage_StableDiffusion,
    "GenerateImage_ComposableDiffusion": GenerateImage_ComposableDiffusion,

    "PreprocessPromptForComposable": PreprocessPromptForComposable,
}

# TODO convert to methods
setupf = lambda n: "Set up pipeline ({})".format(n)
genf = lambda n: "Generate image ({})".format(n)

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetUpPipeline_StableDiffusion": setupf("Stable Diffusion"),

    "GenerateImage_StableDiffusion": genf("Stable Diffusion"),
    "GenerateImage_ComposableDiffusion": genf("Composable Diffusion"),

    "PreprocessPromptForComposable": "Preprocess prompt for Composable Diffusion",
}
