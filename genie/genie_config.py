"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PixelSamplerConfig

from genie.data.dataparsers import GENIEBlenderDataParserConfig
from genie.genie_trainer import GENIETrainerConfig
from genie.genie_model import GENIEModelConfig
from genie.knn.knn_algorithms import TorchKNNConfig, OptixKNNConfig
from genie.utils.schedulers import ChainedSchedulerConfig
from genie.genie_pipeline import GENIEPipelineConfig

MAX_NUM_ITERATIONS = 20000

genie = MethodSpecification(
    config=GENIETrainerConfig(
        method_name="genie",
        steps_per_eval_batch=500,
        steps_per_save=100,
        max_num_iterations=MAX_NUM_ITERATIONS,
        pipeline=GENIEPipelineConfig(
            target_num_samples = 1 << 18,
            datamanager=VanillaDataManagerConfig(
                dataparser=GENIEBlenderDataParserConfig(),
                pixel_sampler=PixelSamplerConfig(
                    rejection_sample_mask=False,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=GENIEModelConfig(
                knn_algorithm=OptixKNNConfig(
                    n_neighbours=16,
                ),
                eval_num_rays_per_chunk=8192,
                near_plane=2.0,
                far_plane=6.0,
                background_color="white",
                disable_scene_contraction=True,
                cone_angle=0.0,
                # near_plane=0.0,
                # far_plane=1e3,
                # background_color="random",
                # disable_scene_contraction=False,
                # cone_angle=1.0 / 256.0,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-06),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
            "log_covs": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ChainedSchedulerConfig(max_steps=MAX_NUM_ITERATIONS),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Gaussian Splatting Encoded Neural Radiance Fields",
)