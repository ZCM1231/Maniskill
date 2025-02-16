import sapien
import numpy as np
import torch
from typing import List
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.sensors.base_sensor import BaseSensorConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils import common, sapien_utils
from copy import deepcopy
@register_agent()
class So100(BaseAgent):
    uid = "So100"
    urdf_path="/home/zcm/lightrobot/ManiSkill/mani_skill/agents/robots/SO100/urdf/SO100.urdf"
    arm_joint_names = [
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
    ]
    gripper_joint_names = [
        "Jaw",
    ]
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0)
        ),
        link=dict(
            Fixed_Jaw=dict(
                material="gripper", patch_radius=0.05, min_patch_radius=0.01
            ),
            Moving_Jaw=dict(
                material="gripper", patch_radius=0.05, min_patch_radius=0.01
            )
        )
    )
    
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e4
    gripper_damping = 1e2
    gripper_force_limit = 200
    def is_static(self, threshold):
        """
        Check if the robot is static by comparing joint velocities with a threshold.
        """
        qvel = self.robot.get_qvel()
        return torch.all(torch.abs(qvel) <= threshold)
    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.5,  # 增加下限以允许更大的开合范围
            upper=1,   # 增加上限以允许更大的开合范围
            stiffness=self.gripper_stiffness * 2,  # 增加刚度以提供更强的抓取力
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit * 1.5,  # 增加力限制以允许更强的抓取
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
              arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 根据 URDF 文件，将 finger1_link 和 finger2_link 分别设置为 Fixed_Jaw 和 Moving_Jaw
        self.finger1_link = None
        self.finger2_link = None
        for link in self.robot.get_links():
            if link.name == "Fixed_Jaw":
                self.finger1_link = link
            elif link.name == "Moving_Jaw":
                self.finger2_link = link
        self.tcp = self.finger1_link
        # 检查是否成功获取夹爪链接
        if self.finger1_link is None or self.finger2_link is None:
            raise ValueError("Failed to find Fixed_Jaw or Moving_Jaw links in the robot.")


    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
    @property
    def _sensor_configs(self) -> List[BaseSensorConfig]:
        return [
            CameraConfig(
                uid="hand_camera",

                pose=Pose.create_from_pq(
                    torch.tensor([0, 0, 0.04], dtype=torch.float32),
                    torch.tensor([0.70710678, 0, 0, -0.70710678], dtype=torch.float32)
                ),
                width=480,
                height=640,
                fov=1.57,  # ~90 degrees
                near=0.01,
                far=100,
                entity_uid="Wrist_Pitch_Roll",
            )
        ]
