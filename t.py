import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
import torch

env = gym.make(
    "PickCubeSO100-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=4,
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos",  # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=1)  # reset with a seed for determinism
done = False
while not np.all(done):  # 检查所有环境是否都完成
    action = env.action_space.sample()

    time.sleep(0.1)
    obs, reward, terminated, truncated, info = env.step(action)
    # 检查是否为 PyTorch 张量，如果是则先移到 CPU 再转换为 numpy 数组
    if isinstance(terminated, torch.Tensor):
        terminated = terminated.cpu().numpy()
    if isinstance(truncated, torch.Tensor):
        truncated = truncated.cpu().numpy()
    done = np.logical_or(terminated, truncated)  # 使用 np.logical_or 进行逐元素逻辑或操作
    env.render()  # a display is required to render
env.close()