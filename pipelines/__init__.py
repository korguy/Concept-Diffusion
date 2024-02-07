from .attend_and_excite_pipeline import AttendAndExcitePipeline
from .composable_diffusion_pipeline import ComposableStableDiffusionPipeline
from .stable_diffusion_pipeline import StableDiffusionPipeline
from .structure_diffusion_pipeline import StructureDiffusionPipeline
from .syntax_guided_generation_pipeline import SynGenDiffusionPipeline

model_dict = {
	'stable_diffusion': StableDiffusionPipeline,
	'composable_diffusion': ComposableStableDiffusionPipeline,
	'structure_diffusion': StructureDiffusionPipeline,
	'attend_and_excite': AttendAndExcitePipeline,
	'syntax_guided_generation': SynGenDiffusionPipeline,
}