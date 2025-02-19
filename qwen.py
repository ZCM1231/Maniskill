
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
image_path = "/home/zcm/Pictures/rgb_image_panda.png"
image = Image.open(image_path)
new_size = (640, 480)  # 调整为 480x640 尺寸，注意 PIL 中顺序是 (width, height)
resized_image = image.resize(new_size, Image.BICUBIC)

model_path = "/home/zcm/qwen_3B"
# 更新消息列表
messages = [
    [{"role": "user", "content": [{"type": "text", "text": "answer in one word where is the cube position relative to the robotic arm in the picture,left or right? "},{"type": "image", "image": resized_image}]}]
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
# case:arm orignal pos cube is left forward
# 	Q:messages = [
# 	    [{"role": "user", "content": [{"type": "image", "image": resized_image}]},{"type": "text", "text": " Suppose you're the white robotic arm in the picture, and the cube is in your direction?"}]
# 	]
# 	A:["system\nYou are a helpful assistant.\nuser\n Suppose you're the white robotic arm in the picture, and the cube is in your direction?
#      \nassistant\nAs the white robotic arm, I am positioned on the wooden surface with my gripper facing towards the red cube. My goal is to pick up the red cube using my gripper.
#       To do this, I need to move my gripper closer to the cube and then apply force to lift it off the surface. Once the cube is lifted, I will release it to complete the task."]


# 	Q:messages = [
# 	    [{"role": "user", "content": [{"type": "text", "text": " Suppose you're the white robotic arm in the picture, and the cube is in your direction?"},{"type": "image", "image": resized_image}]}]
# 	]
# 	A:["system\nYou are a helpful assistant.\nuser\n Suppose you're the white robotic arm in the picture, and the cube is in your direction?
#   \nassistant\nAs the white robotic arm, my task is to pick up the red cube from the table. Here's how I would approach it:\n\n
#   1. **Move Towards the Cube**: First, I need to move towards the red cube on the table. Since the cube is in front of me, I will move forward until I reach it.\n\n
#   2. **Grasp the Cube**: Once I am close enough to the cube, I will extend my gripper mechanism to grasp the cube. The gripper will hold onto the cube securely.\n\n
#   3. **Lift the Cube**: After grasping the cube, I will lift it off the table by applying upward force with my gripper. This will ensure that the cube is lifted safely without damaging it or the table.\n\n
#   4. **Move Away from the Table**: Finally, I will move away from the table while holding the cube. This will allow me to transport the cube to its desired location.\n\nBy 
#   following these steps, I can successfully pick up the red cube and move it as needed."]
#     Q:messages = [
#     [{"role": "user", "content": [{"type": "image", "image": resized_image},{"type": "text", "text": " where is the cube position relative to the robotic arm in the picture"}]}]
# ]
#     A:['system\nYou are a helpful assistant.\nuser\n where is the cube position relative to the robotic arm in the picture\n
# assistant\nIn the image, the red cube is positioned to the left of the robotic arm. The robotic arm is located towards the center-right of the image, and the red cube is slightly to its left.']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " where is the cube position relative to the robotic arm in the picture"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n where is the cube position relative to the robotic arm in the picture\n
#    assistant\nThe red cube is positioned to the left of the robotic arm on the wooden surface. The robotic arm is facing towards the right side of the image, and the red cube is slightly ahead of it, closer to the bottom edge of the image.']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture\nassistant\nBelow']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,left or right?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nLeft']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,forward or backward?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nForward']


# case:arm orignal pos cube is left backward

# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,forward or backward?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nForward']

# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,left or right?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nLeft']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " where is the cube position relative to the robotic arm in the picture"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n where is the cube position relative to the robotic arm in the picture\n
#  assistant\nThe red cube is positioned directly below the robotic arm on the wooden surface. The robotic arm appears to be hovering above the cube, with its end effector (the part that would typically grasp or manipulate objects) aligned with the top of the cube.']






# case:arm orignal pos cube is right forward
# 	Q:messages = [
# 	    [{"role": "user", "content": [{"type": "text", "text": " Suppose you're the white robotic arm in the picture, and the cube is in your direction?"},{"type": "image", "image": resized_image}]}]
# 	]
#   A:["system\nYou are a helpful assistant.\nuser\n Suppose you're the white robotic arm in the picture, and the cube is in your direction?
#   \nassistant\nAs the white robotic arm, my task is to pick up the red cube from the table. Here's how I would approach it:\n\n
#   1. **Move Towards the Cube**: First, I need to move towards the red cube on the table. Since the cube is in front of me, I will move forward until I reach it.\n\n
#   2. **Grasp the Cube**: Once I am close enough to the cube, I will use my gripper mechanism to grasp it. The gripper will hold onto the cube securely.\n\n
#   3. **Lift the Cube**: With the cube securely grasped, I will lift it off the table. This involves moving my arm upwards while keeping the cube in my grip.\n\n
#   4. **Move Away from the Table**: After lifting the cube, I will move away from the table to place it somewhere else or return it to its original position if necessary.\n\n
#   By following these steps, I can successfully pick up the red cube and perform any further actions required with it."]
#    Q:messages = [
#    [{"role": "user", "content": [{"type": "image", "image": resized_image},{"type": "text", "text": " Suppose you're the white robotic arm in the picture, and the cube is in your direction?"}]}]
#     ]
#    A:["system\nYou are a helpful assistant.\nuser\n Suppose you're the white robotic arm in the picture, and the cube is in your direction?
# \nassistant\nAs the white robotic arm, I am positioned on the wooden surface with my gripper facing towards the red cube. My goal is to grasp the cube and move it to a new location."]
# Q:messages = [
#     [{"role": "user", "content": [{"type": "image", "image": resized_image},{"type": "text", "text": " where is the cube position relative to the robotic arm in the picture"}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n where is the cube position relative to the robotic arm in the picture\n
#    assistant\nIn the image, the red cube is positioned to the right of the robotic arm. The robotic arm is located on the left side of the image, and the red cube is on the right side.']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " where is the cube position relative to the robotic arm in the picture"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n where is the cube position relative to the robotic arm in the picture\n
#    assistant\nIn the image, the robotic arm is positioned on the left side of the table, and the red cube is located on the right side of the table. The robotic arm appears to be facing towards the red cube.']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture\nassistant\nBelow']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,left or right?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nRight']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,forward or backward?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nForward']





# case:arm orignal pos cube is right backward

# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,forward or backward?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nForward']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " answer in one word where is the cube position relative to the robotic arm in the picture,left or right?"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n answer in one word where is the cube position relative to the robotic arm in the picture,left or right?\nassistant\nLeft']
# Q:messages = [
#     [{"role": "user", "content": [{"type": "text", "text": " where is the cube position relative to the robotic arm in the picture"},{"type": "image", "image": resized_image}]}]
# ]
# A:['system\nYou are a helpful assistant.\nuser\n where is the cube position relative to the robotic arm in the picture\n
#    assistant\nThe cube is positioned to the right of the robotic arm on the wooden surface. The robotic arm is located more towards the center-left of the image, while the red cube is situated further to the right.']