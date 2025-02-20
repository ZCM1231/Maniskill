import gymnasium as gym
import mani_skill.envs
import numpy as np
import os
import torch
from torchvision.utils import save_image

save_dir = "/home/zcm/Pictures"  # 定义图像保存目录

last_qpos_path = os.path.join(save_dir, 'last_qpos.npy')
if os.path.exists(last_qpos_path):
    desired_qpos = np.load(last_qpos_path)
    print('read-----------------------------------qpos')
else:

    desired_qpos= np.array(
                    [
                        0.0,
                        np.pi / 8,
                        0,
                        -np.pi * 5 / 8,
                        0,
                        np.pi * 3 / 4,
                        np.pi / 4,
                        0.04,
                        0.04,
                    ])

# 创建环境
env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    render_mode="human",    
    robot_init_qpos=desired_qpos, 
    cube_position=[0, -0.2],
    cube_rotation=[1, 0, 0, 0]
    

)

print("Observation space", env.observation_space)
print("Action space", env.action_space)

# 重置环境
obs, _ = env.reset(seed=0)
done = False

# 定义位置增量，这里简单示例，每次沿 x 轴正向移动 0.01
delta_pos = np.array([0, 0, 0.1])

# 控制抓取的维度，0 表示不抓取，1 表示抓取，这里先设为 0
gripper_action = np.array([1])

while not done:
    # 构建 ee_delta_pos 控制模式下的动作
    # 动作是 3 维位置增量和 1 维抓取控制的组合
    action = np.concatenate([delta_pos, gripper_action])
    # delta_pos=np.array([0,0,0])
    # 调整动作形状为 (1, 4)，因为 num_envs = 1，每个环境的动作是 4 维的
    action = np.expand_dims(action, axis=0)

    # 确保动作在动作空间范围内
    # action = np.clip(action, env.action_space.low, env.action_space.high)

    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs)
    rgb_image = obs['sensor_data']['base_camera']['rgb']
    # 由于图像数据通常为 (C, H, W, 3) 格式（C 为图像数量，这里通常为 1），如果 C 为 1 则去掉该维度
    if rgb_image.ndim == 4 and rgb_image.shape[0] == 1:
        rgb_image = rgb_image.squeeze(0)
    # 将数据类型转换为 float 并归一化到 [0, 1] 范围
    rgb_image = rgb_image.float() / 255.0
    # 调整通道顺序，从 (H, W, C) 到 (C, H, W)
    rgb_image = rgb_image.permute(2, 0, 1)

    # 保存 RGB 图像
    rgb_image_path = os.path.join(save_dir, 'rgb_image_panda.png')
    save_image(rgb_image, rgb_image_path)
    print('saved rgb picture')
    last_qpos=obs['agent']['qpos']
    np.save(last_qpos_path, last_qpos)
    print(last_qpos)
    done = terminated or truncated
    env.render()

# 关闭环境
env.close()