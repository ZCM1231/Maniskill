import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
import torch
from lerobot.feetech_arm import feetech_arm
import json
control_freq = 30  # Hz
control_period = 1.0 / control_freq

def convert_step_to_rad(steps):
    """Convert motor steps to rad.
    4096 steps = 6.28 rad (full rotation)
    """
    return steps * (6.28 / 4096.0)

def load_joint_offsets():
    """Load joint offsets from calibration file and convert to degrees."""
    with open("/home/zcm/lightrobot/ManiSkill/lerobot/right_follower.json", 'r') as f:
        calibration = json.load(f)
    # Convert start positions from steps to degrees
    start_pos_deg = [convert_step_to_rad(pos) for pos in calibration['start_pos']]
    return start_pos_deg

env = gym.make(
    "PickCubeSO100-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos",  # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    cube_position=[-0.45, 0.2],  # 设置立方体的x和y坐标
    cube_rotation=[1, 0, 0, 0]  # 设置立方体的旋转(这是一个四元数,表示无旋转)
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=1)  # reset with a seed for determinism
done = False
arm = feetech_arm(driver_port="/dev/ttyACM0", calibration_file="/home/zcm/lightrobot/ManiSkill/lerobot/right_follower.json")
print("Hardware initialized")

# Load joint offsets from calibration
joint_offset_real = load_joint_offsets()
print("Joint offsets (steps):", joint_offset_real)
# Initialize joint offsets
joint_offset_sim = [3.0280669, 7.147486, -1.0413924, 3.1595317, 5.025189, 0.6514962]
# [0, -2.14, 1.83, -0.40, 1.83, -0.1]-[-3.0280669 -9.087486   2.9713924 -3.5595317 -3.195189  -1.2514962]
joint_offset_real = load_joint_offsets()
print(joint_offset_real)
# print(joint_offset_sim)
done = False
while not done:  # 检查环境是否完成
    start_time = time.time() 
    # Read joint positions from hardware
    joint_positions = arm.feedback()
    # Convert joint positions considering offsets
    joint_positions = np.array(joint_positions)
    
    # Apply offsets
    for i in range(len(joint_positions)):
        joint_positions[i] = joint_positions[i]-joint_offset_real[i]+joint_offset_sim[i]              # + joint_offset_sim[i]
    
    # 更新关节位置,包括夹爪
    action = joint_positions
    

    end_time = time.time() 
    elapsed_time = end_time - start_time  
    print(f"延时时间: {elapsed_time:.6f} 秒") 
    
    print(f'Action: {action}')
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 打印观察空间和动作空间的信息
    print("Observation space shape:", obs.shape)
    print("Action space shape:", env.action_space.shape)
    print("Reward:", reward)
    print("Info:", info)
    
    # 检查是否为 PyTorch 张量,如果是则先移到 CPU 再转换为 numpy 数组
    if isinstance(terminated, torch.Tensor):
        terminated = terminated.cpu().numpy()
    if isinstance(truncated, torch.Tensor):
        truncated = truncated.cpu().numpy()
    done = terminated or truncated  # 使用逻辑或操作
    env.render()  # a display is required to render
    elapsed = time.time() - start_time
    if elapsed < control_period:
        time.sleep(control_period - elapsed)
env.close()
