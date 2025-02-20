# -*- coding: utf-8 -*-
# flake8: noqa

from qiniu import Auth, put_file, etag
import qiniu.config
from qiniu import Auth
from qiniu import BucketManager

#需要填写你的 Access Key 和 Secret Key
access_key = 'BziER-3xrxPOXEqYONtP7ZQqEsxKVDWm7ynSh-sF'
secret_key = 'm6FkHnB_iBNfXhZ8jIN1tbpDaPZtZo_e6gXqSXJ9'
#构建鉴权对象
q = Auth(access_key, secret_key)
#要上传的空间
bucket_name = 'vlm-save-space'
#要上传文件的本地路径
localfile = '/home/zcm/Pictures/rgb_image_panda.png'



#初始化BucketManager
bucket = BucketManager(q)
import time
import gymnasium as gym
import mani_skill.envs
import numpy as np
import os
import torch
from torchvision.utils import save_image

save_dir = "/home/zcm/Pictures"  # 定义图像保存目录

from openai import OpenAI
import numpy as np
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

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
    cube_position=[0, -0.3],
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
upload_cnt=0
import time

# 获取当前时间的时间戳
timestamp = time.time()
# 将时间戳转换为本地时间的结构化对象
local_time = time.localtime(timestamp)
# 格式化输出当前时间
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
while not done:
    upload_cnt+=1
    #上传后保存的文件名
    qiniu_key = f'{formatted_time}rgb_image_panda{upload_cnt}.png'
    #生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, qiniu_key, 3600)
    url=f"http://srxiyn0vd.hn-bkt.clouddn.com/{formatted_time}rgb_image_panda{upload_cnt}.png"
    print(url)
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

    #图床更新

    ret, info = put_file(token, qiniu_key, localfile, version='v2')
    print(info)
    assert ret['key'] == qiniu_key
    assert ret['hash'] == etag(localfile)

    # 保存 RGB 图像
    rgb_image_path = os.path.join(save_dir, 'rgb_image_panda.png')
    save_image(rgb_image, rgb_image_path)
    print('saved rgb picture')
    last_qpos=obs['agent']['qpos']
    np.save(last_qpos_path, last_qpos)
    # print(last_qpos)
    done = terminated or truncated
    # env.render()




    completion = client.chat.completions.create(
    extra_headers={
        
    },
    extra_body={},
    model="openai/gpt-4o-2024-11-20",
    # model="qwen/qwen-max",
    #/qwen/qwen2.5-vl-72b-instruct:free
    #qwen/qwen-2.5-7b-instruct
    #openai/gpt-4o-2024-11-20
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "The table has robot who have blue gripper and a red cube as shown in the image. Tell me the direction of the cube respect to the robot.the perspective is front just give one word for answer from the perspective of picture : left or right? only give me final answer"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": url
            }
            }
        ]
        }
    ]
    )

# 检查 completion.choices 是否为 None
    if completion.choices is not None:
        # 获取 API 返回的消息内容
        content = completion.choices[0].message.content
        # print(content)

        # 转换为小写以便不区分大小写进行查找
        response = content.lower()

        delta_pos = np.array([0, 0, 0])  # 默认值

        if "left" in response:
            print(f"\n\n\n\nThe cube is on the left of the robotic arm.\n\n\n\n{response}\n\n\n\n")
            delta_pos = np.array([0, -.5,0])
        elif "right" in response:
            print(f"\n\n\n\nThe cube is on the right of the robotic arm.\n\n\n\n{response}\n\n\n\n")
            delta_pos = np.array([0, 0.5, 0])
        else:
            print(f"\n\n\n\nCould not determine position from: {response}\n\n\n\n")
            print(completion.choices[0].message)

        # 可以在这里使用 delta_pos 进行后续操作
        print("最终的 delta_pos:", delta_pos)
    else:
        print("API 返回的 choices 为空。")
        print("完整的 API 返回内容：", completion)


    # # 删除bucket_name 中的文件 key
    # ret, info = bucket.delete(bucket_name, qiniu_key)
    # print(info)

# 关闭环境
env.close()




# http://srxiyn0vd.hn-bkt.clouddn.com/rgb_image_panda.png


