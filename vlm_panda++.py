import gymnasium as gym
import mani_skill.envs
import numpy as np
import os
import torch
from torchvision.utils import save_image
from qwen_vl_utils import process_vision_info
from PIL import Image


# 设置环境变量
os.environ["VIDEO_MAX_PIXELS"] = str(int(32000 * 28 * 28 * 0.9))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model_path = "/home/zcm/qwen_3B"
# 更新消息列表
save_dir = "/home/zcm/Pictures"  # 定义图像保存目录

last_qpos_path = os.path.join(save_dir, 'last_qpos.npy')
if os.path.exists(last_qpos_path):
    desired_qpos = np.load(last_qpos_path)
    print('read-----------------------------------qpos')
else:
    desired_qpos = np.array(
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
        ]
    )

# 定义位置增量，这里简单示例，每次沿 z 轴正向移动 0.1
delta_pos = np.array([0, 0, 0])
# 控制抓取的维度，0 表示不抓取，1 表示抓取，这里先设为 0
gripper_action = np.array([1])

done = False
while not done:
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

    # 构建 ee_delta_pos 控制模式下的动作
    # 动作是 3 维位置增量和 1 维抓取控制的组合
    action = np.concatenate([delta_pos, gripper_action])
    # 调整动作形状为 (1, 4)，因为 num_envs = 1，每个环境的动作是 4 维的
    action = np.expand_dims(action, axis=0)

    try:
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
    except AttributeError as e:
        print(f"Error: {e}. Reinitializing the environment.")
        # 重新创建和重置环境
        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pos",
            sensor_configs=dict(shader_pack="rt"),  # 为所有传感器相机设置"rt"着色器包
            render_mode="human",
            robot_init_qpos=desired_qpos,
        )
        obs, _ = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)

    # 处理 RGB 图像
    rgb_image = obs['sensor_data']['base_camera']['rgb']
    if rgb_image.ndim == 4 and rgb_image.shape[0] == 1:
        rgb_image = rgb_image.squeeze(0)
    rgb_image = rgb_image.float() / 255.0
    rgb_image = rgb_image.permute(2, 0, 1)

    # 保存 RGB 图像
    rgb_image_path = os.path.join(save_dir, 'rgb_image_panda.png')
    save_image(rgb_image, rgb_image_path)
    print('saved rgb picture')

    last_qpos = obs['agent']['qpos']
    np.save(last_qpos_path, last_qpos)
    # print(last_qpos)

    done = terminated or truncated
    env.close()  # 关闭环境
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # 运行 VLM
    # 打开图像并调整大小
    image_path = "/home/zcm/Pictures/rgb_image_panda.png"
    gripper_Image_path="/home/zcm/Pictures/gripper.png"
    gripper_image=Image.open(gripper_Image_path)
    image = Image.open(image_path)
    new_size = (640, 480)  # 调整为 480x640 尺寸，注意 PIL 中顺序是 (width, height)
    resized_image = image.resize(new_size, Image.BICUBIC)

    messages = [
    [{"role": "user", "content": [{"type": "image", "image": gripper_image},{"type": "text", "text": "this is how the gripper looks like,you should take middle of it as the position of gripper"},{"type": "text", "text": "You're at a wooden table, and across from you is a robotic arm and a cube. where the cube is relative to the arm gripper, left or right?"}, {"type": "image", "image": resized_image}]}]
    ]
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.gradient_checkpointing_enable()  # 启用梯度检查点

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
    # print(inputs)

    # 将输入张量移动到和模型相同的设备上
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 调整生成参数
    generated_ids = model.generate(**inputs, max_length=4096)
    # print(generated_ids)

    # 解码生成的 ID 为文本
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # print(generated_text)
    # 判断 generated_text 中 assistant\n 后面是 Left 还是 Right
    response = ""
    for text in generated_text:
        index = text.find("assistant\n")
        if index != -1:
            response = text[index + len("assistant\n"):].strip().lower()  # 转换为小写
            break

    delta_pos = np.array([0, 0, 0])  # 默认值
    
    if "left" in response:
        print(f"\n\n\n\nThe cube is on the left of the robotic arm.\n\n\n\n{response}\n\n\n\n")
        delta_pos = np.array([0, -1, 0])
    elif "right" in response:
        print(f"\n\n\n\nThe cube is on the right of the robotic arm.\n\n\n\n{response}\n\n\n\n")
        delta_pos = np.array([0, 1, 0])
    else:
        print(f"\n\n\n\nCould not determine position from: {response}\n\n\n\n")

    # 更新 desired_qpos 为 last_qpos
    desired_qpos = last_qpos

    # 释放 VLM 相关的内存
    del model, processor, inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()