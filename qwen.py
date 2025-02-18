
# messages = [
#     # Image
#     ## Local file path
#     [{"role": "user", "content": [{"type": "image", "image": "/home/zcm/pp5_100.png"}, {"type": "text", "text": "Describe this image."}]}],
#     # ## Image URL
#     # [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
#     # ## Base64 encoded image
#     # [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}],
#     # ## PIL.Image.Image
#     # [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "Describe this image."}]}],
#     # ## Model dynamically adjusts image size, specify dimensions if required.
#     # [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}],
#     # # Video
#     # ## Local video path
#     # [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4"}, {"type": "text", "text": "Describe this video."}]}],
#     # ## Local video frames
#     # [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/extracted_frame1.jpg", "file:///path/to/extracted_frame2.jpg", "file:///path/to/extracted_frame3.jpg"],}, {"type": "text", "text": "Describe this video."},],}],
#     # ## Model dynamically adjusts video nframes, video height and width. specify args if required.
#     # [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", "fps": 2.0, "resized_height": 280, "resized_width": 280}, {"type": "text", "text": "Describe this video."}]}],
# ]

import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# 设置环境变量
os.environ["VIDEO_MAX_PIXELS"] = str(int(32000 * 28 * 28 * 0.9))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 打开图像并调整大小
image_path = "/home/zcm/Pictures/rgb_image.png"
image = Image.open(image_path)
new_size = (640, 480)  # 调整为 480x640 尺寸，注意 PIL 中顺序是 (width, height)
resized_image = image.resize(new_size, Image.BICUBIC)

model_path = "/home/zcm/qwen_3B"
# 更新消息列表
messages = [
    [{"role": "user", "content": [{"type": "image", "image": resized_image}, {"type": "text", "text": "where is the cube ? Suppose you are the white 6DOF robot arm with griper in the picture, and the view Angle of the picture is top-down. Your task is to pick up the block and lift it 10cm high, and the side length of the block is 2cm. Each time you perform an action, you are only allowed to move 0.1cm up, down, left, right ,back, and forth in one direction"}]}]
]
# Suppose you are the white 6DOF robot arm with claws in the picture, and the view Angle of the picture is top-down. Your task is to pick up the block and lift it 10cm high, and the side length of the block is 2cm. Each time you perform an action, you are only allowed to move 0.1cm up, down, left, right ,back, and forth in one direction
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