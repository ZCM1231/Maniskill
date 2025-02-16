import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.utils.sapien_utils import look_at
from mani_skill.sensors.camera import CameraConfig

@register_env("SimpleCubeEnv-v0", max_episode_steps=500)
class SimpleCubeEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        self.cube_half_size = 0.02
        super().__init__(*args, robot_uids="none", **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[-0.6, 0, 0.3], target=[-0.35, 0, 0])
        return [CameraConfig("base_camera", pose, 480, 640, np.pi / 2, 0.01, 100)]

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        
        self.cube = actors.build_colorful_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            self.cube.set_pose(sapien.Pose(p=xyz[0].cpu().numpy()))

    def _get_obs_extra(self, info: dict):
        return dict(
            cube_pose=self.cube.pose.raw_pose,
        )

    def _get_obs_agent(self):
        return {}  # No agent, so return an empty dict

    def evaluate(self):
        return {}

    def get_reward(self, obs: dict, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, info
