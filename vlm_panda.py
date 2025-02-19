import gymnasium as gym
import mani_skill.envs
import numpy as np
import os
import torch
from torchvision.utils import save_image
import os
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
# 设置环境变量
os.environ["VIDEO_MAX_PIXELS"] = str(int(32000 * 28 * 28 * 0.9))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 打开图像并调整大小
image_path = "/home/zcm/Pictures/rgb_image_panda.png"
image = Image.open(image_path)
new_size = (640, 480)  # 调整为 480x640 尺寸，注意 PIL 中顺序是 (width, height)
resized_image = image.resize(new_size, Image.BICUBIC)

model_path = "/home/zcm/qwen_3B"
# 更新消息列表
messages = [
    [{"role": "user", "content": [{"type": "text", "text": "answer in one word where is the cube position relative to the robotic arm in the picture,left or right? "},{"type": "image", "image": resized_image}]}]
]
save_dir = "/home/zcm/Pictures"  # 定义图像保存目录
# 创建环境
env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    render_mode="human",    
    cube_position=[0, -0.2],
    cube_rotation=[1, 0, 0, 0]
    

)

print("Observation space", env.observation_space)
print("Action space", env.action_space)

# 重置环境
obs, _ = env.reset(seed=0)
done = False

# 定义位置增量，这里简单示例，每次沿 x 轴正向移动 0.01
delta_pos = np.array([0, 0.1, 0])

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

    done = terminated or truncated
    env.render()
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.gradient_checkpointing_enable()  # 启用梯度检查点

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    print(inputs)

    # 将输入张量移动到和模型相同的设备上
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 调整生成参数
    generated_ids = model.generate(**inputs, max_length=2048)
    print(generated_ids)

    # 解码生成的 ID 为文本
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_text)
# 关闭环境
env.close()