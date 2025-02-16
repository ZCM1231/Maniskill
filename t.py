import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
import torch
from lerobot.feetech_arm import feetech_arm
import json
import os
import cv2
from datetime import datetime

# 设置控制频率和保存路径
control_freq = 30  # Hz
control_period = 1.0 / control_freq
save_dir = "/home/zcm/Pictures"  # 定义图像保存目录
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

def save_camera_images(obs, folder_name="capture"):
    """
    保存相机图像到指定目录
    参数:
        obs - 从环境获取的观测数据
        folder_name - 每次捕获创建的子目录名称前缀
    """    
    # 创建带时间戳的唯一子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_dir = os.path.join(save_dir, f"{folder_name}_{timestamp}")
    os.makedirs(capture_dir, exist_ok=True)
    
    # 遍历所有相机
    for cam_name in ['c_camera', 'cube_camera', 'base_camera', 'hand_camera']:
        try:
            # 获取RGB图像张量 (形状为 [1, H, W, 3])
            rgb_tensor = obs['sensor_data'][cam_name]['rgb']
            
            # 转换为numpy数组并去掉批次维度
            rgb_array = rgb_tensor.squeeze(0).cpu().numpy()
            
            # 转换为OpenCV所需的BGR格式
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            # 构造文件名并保存
            filename = os.path.join(capture_dir, f"{cam_name}.png")
            cv2.imwrite(filename, bgr_array)
            
        except KeyError as e:
            print(f"Warning: Cam {cam_name} not found in observation - {e}")
        except Exception as e:
            print(f"Error saving {cam_name} image: {str(e)}")

def convert_step_to_rad(steps):
    """Convert motor steps to rad."""
    return steps * (6.28 / 4096.0)

def load_joint_offsets():
    """Load joint offsets from calibration file."""
    with open("/home/zcm/lightrobot/ManiSkill/lerobot/right_follower.json", 'r') as f:
        calibration = json.load(f)
    return [convert_step_to_rad(pos) for pos in calibration['start_pos']]

# 初始化环境
env = gym.make(
    "PickCubeSO100-v1",
    num_envs=1,
    obs_mode="rgbd",  # 确保包含相机数据
    control_mode="pd_joint_pos",
    render_mode="human",
    human_render_camera_configs=dict(shader_pack="rt"),
    sensor_configs=dict(shader_pack="rt"),  # 为所有传感器相机设置"rt"着色器包
    cube_position=[-0.45, 0.2],
    cube_rotation=[1, 0, 0, 0]
)
print("Environment initialized")

# 硬件初始化
arm = feetech_arm(
    driver_port="/dev/ttyACM0",
    calibration_file="/home/zcm/lightrobot/ManiSkill/lerobot/right_follower.json"
)
print("Hardware connection established")

# 载入标定参数
joint_offset_real = load_joint_offsets()
joint_offset_sim = [3.0280669, 7.147486, -1.0413924, 3.1595317, 5.025189, 0.6514962]

# 主控制循环
obs, _ = env.reset(seed=1)
done = False

while not done:
    cycle_start = time.time()
    
    # 1. 读取硬件关节位置
    raw_joints = np.array(arm.feedback())
    
    # 2. 应用标定转换
    calibrated_joints = raw_joints - joint_offset_real + joint_offset_sim
    
    # 3. 构建动作向量
    action = torch.tensor(calibrated_joints, dtype=torch.float32)  # 转换为PyTorch张量
    
    # 4. 执行环境步进
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 5. 保存相机图像 (新增部分)
    save_camera_images(obs, "step_capture")
    
    # 6. 打印调试信息
    print(f"Action: {action.numpy()}")
    print(f"Info: {info}")
    
    # 7. 更新终止状态
    done = terminated or truncated
    
    # 8. 频率控制
    env.render()  # a display is required to render
    elapsed = time.time() - cycle_start
    if elapsed < control_period:
        time.sleep(control_period - elapsed)

# 关闭资源
env.close()
print("All resources released")
