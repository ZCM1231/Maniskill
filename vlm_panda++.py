import gymnasium as gym
import mani_skill.envs
import numpy as np
import os
import torch
from torchvision.utils import save_image
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 配置初始化 ==============================================================
# 环境变量设置
os.environ["VIDEO_MAX_PIXELS"] = str(int(32000 * 28 * 28 * 0.9))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 模型路径配置
model_path = "/home/zcm/qwen_3B"
save_dir = "/home/zcm/Pictures"

# 硬件资源检查 ============================================================
assert torch.cuda.is_available(), "需要CUDA GPU支持"
print(f"初始显存: {torch.cuda.mem_get_info()[1]/1024**2:.2f}MB")

# 模型预加载 ==============================================================
# 单例模式加载模型（关键修改点）
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()  # 固定为评估模式
model.gradient_checkpointing_enable()

# 环境初始化 ==============================================================
env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    render_mode="human",    
    cube_position=[0, -0.2],
    cube_rotation=[1, 0, 0, 0]
)
obs, _ = env.reset(seed=0)

# 控制参数 ================================================================
delta_pos = np.array([0, 0.1, 0])
gripper_action = np.array([1])

# 主循环 ================================================================
with torch.inference_mode():  # 全局禁用梯度计算
    done = False
    while not done:
        # 环境交互部分
        action = np.expand_dims(np.concatenate([delta_pos, gripper_action]), axis=0)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 图像处理部分
        rgb_tensor = obs['sensor_data']['base_camera']['rgb']
        if rgb_tensor.ndim == 4 and rgb_tensor.shape[0] == 1:
            rgb_tensor = rgb_tensor.squeeze(0)
        rgb_tensor = (rgb_tensor.float() / 255.0).permute(2, 0, 1)
        save_image(rgb_tensor, os.path.join(save_dir, 'rgb_image_panda.png'))
        
        # 模型推理部分
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "answer in one word where is the cube position relative to the robotic arm in the picture,left or right? "},
                {"type": "image", "image": Image.open(os.path.join(save_dir, 'rgb_image_panda.png'))}
            ]
        }]
        
        # 内存优化处理流程
        with torch.cuda.amp.autocast():  # 混合精度
            inputs = processor(
                text=processor.apply_chat_template(messages, tokenize=False),
                images=process_vision_info(messages)[0],
                return_tensors="pt"
            ).to(model.device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,  # 限制生成长度
                early_stopping=True
            )
            
            # 显式释放中间变量
            del inputs
            torch.cuda.empty_cache()
            
            # 结果解码
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"推理结果: {generated_text[0]}")
            
            # 显式释放生成结果
            del generated_ids, generated_text
            torch.cuda.empty_cache()
        
        env.render()

env.close()
