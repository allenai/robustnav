"""
Experiment config to train a ObjectNav RGB policy
"""

# Required imports
import glob
import os
from abc import ABC
from math import ceil
from typing import Dict, Any, List, Optional, Sequence, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor
from allenact.base_abstractions.task import TaskSampler

from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.utils.system import get_logger
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)

from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.ithor_plugin.ithor_util import horizontal_to_vertical_fov
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor

from allenact_plugins.robothor_plugin.robothor_task_samplers import (
    ObjectNavDatasetTaskSampler,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask

from allenact.embodiedai.preprocessors.custom import CustomPreprocessor

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig

from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)

from allenact.base_abstractions.sensor import DepthSensor, RGBSensor


class ObjectNavS2SRGBCustomDDPPO(ExperimentConfig, ABC):
    """A ObjectNav Experiment Config using RGB sensors and DDPPO"""

    def __init__(self):
        super().__init__()

        # Task Parameters
        self.ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

        self.STEP_SIZE = 0.25
        self.ROTATION_DEGREES = 30.0
        self.DISTANCE_TO_GOAL = 0.2
        self.VISIBILITY_DISTANCE = 1.0
        self.STOCHASTIC = True
        self.HORIZONTAL_FIELD_OF_VIEW = 79

        self.CAMERA_WIDTH = 400
        self.CAMERA_HEIGHT = 300
        self.SCREEN_SIZE = 224
        self.MAX_STEPS = 500

        # Random crop specifications for data augmentations
        self.CROP_WIDTH = 320
        self.CROP_HEIGHT = 240

        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "reached_max_steps_reward": 0.0,
            "shaping_weight": 1.0,
        }

        self.NUM_PROCESSES = 60

        self.TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
        self.SAMPLER_GPU_IDS = self.TRAIN_GPU_IDS
        self.VALID_GPU_IDS = [torch.cuda.device_count() - 1]
        self.TEST_GPU_IDS = [torch.cuda.device_count() - 1]
        # Preprecessors should be empty for custom config as model is chosen and created later
        self.PREPROCESSORS = list()

        self.TARGET_TYPES = tuple(
            sorted(
                [
                    "AlarmClock",
                    "Apple",
                    "BaseballBat",
                    "BasketBall",
                    "Bowl",
                    "GarbageCan",
                    "HousePlant",
                    "Laptop",
                    "Mug",
                    "SprayBottle",
                    "Television",
                    "Vase",
                ]
            )
        )

        OBSERVATIONS = [
            "rgb_resnet",
            "goal_object_type_ind",
        ]

        self.ENV_ARGS = dict(
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT,
            continuousMode=True,
            applyActionNoise=self.STOCHASTIC,
            agentType="stochastic",
            rotateStepDegrees=self.ROTATION_DEGREES,
            visibilityDistance=self.VISIBILITY_DISTANCE,
            gridSize=self.STEP_SIZE,
            snapToGrid=False,
            agentMode="locobot",
            fieldOfView=horizontal_to_vertical_fov(
                horizontal_fov_in_degrees=self.HORIZONTAL_FIELD_OF_VIEW,
                width=self.CAMERA_WIDTH,
                height=self.CAMERA_HEIGHT,
            ),
            include_private_scenes=False,
            renderDepthImage=False,
        )

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO"

    def monkey_patch_datasets(self, train_dataset, val_dataset, test_dataset):
        if train_dataset is not None:
            self.TRAIN_DATASET_DIR = os.path.join(os.getcwd(), train_dataset)
        else:
            self.TRAIN_DATASET_DIR = os.path.join(
                os.getcwd(), "datasets/robothor-objectnav/train"
            )

        if val_dataset is not None:
            self.VAL_DATASET_DIR = os.path.join(os.getcwd(), val_dataset)
        else:
            self.VAL_DATASET_DIR = os.path.join(
                os.getcwd(), "datasets/robothor-objectnav/robustnav_eval"
            )

        if test_dataset is not None:
            self.TEST_DATASET_DIR = os.path.join(os.getcwd(), test_dataset)
        else:
            self.TEST_DATASET_DIR = os.path.join(
                os.getcwd(), "datasets/robothor-objectnav/robustnav_eval"
            )

    def monkey_patch_sensor(
        self,
        corruptions=None,
        severities=None,
        random_crop=False,
        color_jitter=False,
        random_shift=False,
    ):
        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
                corruptions=corruptions,
                severities=severities,
                random_crop=random_crop,
                random_translate=random_shift,
                crop_height=self.CROP_HEIGHT,
                crop_width=self.CROP_WIDTH,
                color_jitter=color_jitter,
            ),
            GoalObjectTypeThorSensor(object_types=self.TARGET_TYPES,),
        ]

    # DD-PPO Base
    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000 if torch.cuda.is_available() else 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def create_model(self, **kwargs) -> nn.Module:
        rgb_uuid = "rgb_resnet"
        goal_sensor_uuid = "goal_object_type_ind"

        if '64' in self.model_name:
            latent_sz = 32
        else:
            raise Exception(f"{self.model_name} is not a properly named model should end with \'_input size\'")

        # TODO: Change this to custom actor critic
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid=rgb_uuid,
            hidden_size=512,
            goal_dims=32,
        )
    
    def create_preprocessor(self, model_name, ckpt_path, encoder_base):
        self.model_name = model_name
        self.PREPROCESSORS = [
            Builder(
                CustomPreprocessor,
                {
                    "model_name": model_name,
                    "ckpt_path": ckpt_path,
                    "encoder_base": encoder_base,
                    "input_height": self.SCREEN_SIZE,
                    "input_width": self.SCREEN_SIZE,
                    "input_uuids": [f"{model_name}_lowres"],
                    "output_uuid": f"rgb_{model_name}",
                },
            ),
        ]

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[int] = []
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAIN_GPU_IDS * workers_per_device
            )
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else evenly_distribute_count_into_bins(self.NUM_PROCESSES, len(gpu_ids))
            )
            sampler_devices = self.SAMPLER_GPU_IDS
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
        elif mode == "test":
            nprocesses = 15 if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=gpu_ids,
            sampler_devices=sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavDatasetTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
        include_expert_sensor: bool = True,
        allow_oversample: bool = False,
    ) -> Dict[str, Any]:
        path = os.path.join(scenes_dir, "*.json.gz")
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if len(scenes) == 0:
            raise RuntimeError(
                (
                    "Could find no scene dataset information in directory {}."
                    " Are you sure you've downloaded them? "
                    " If not, see https://allenact.org/installation/download-datasets/ information"
                    " on how this can be done."
                ).format(scenes_dir)
            )

        oversample_warning = (
            f"Warning: oversampling some of the scenes ({scenes}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(scenes)`"
                    f" ({total_processes} > {len(scenes)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(scenes) != 0:
                get_logger().warning(oversample_warning)
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        elif len(scenes) % total_processes != 0:
            get_logger().warning(oversample_warning)

        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": [
                s
                for s in self.SENSORS
                if (include_expert_sensor or not isinstance(s, ExpertActionSensor))
            ],
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
            "env_args": {
                **self.ENV_ARGS,
                "x_display": (
                    f"0.{devices[process_ind % len(devices)]}"
                    if devices is not None
                    and len(devices) > 0
                    and devices[process_ind % len(devices)] >= 0
                    else None
                ),
            },
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            scenes_dir=os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            allow_oversample=True,
        )
        res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = True
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            scenes_dir=os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
            allow_oversample=False,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            scenes_dir=os.path.join(self.TEST_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
            allow_oversample=False,
        )
        res["scene_directory"] = self.TEST_DATASET_DIR
        res["loop_dataset"] = False
        return res
