import pandas as pd
import re

# 定义原始结果字符串
result_text = """
Running benchmark for env_id: FrankaPickCubeBenchmark-v1, num_envs: 4
# -------------------------------------------------------------------------- #
Task ID: FrankaPickCubeBenchmark-v1, 4 parallel environments, sim_backend=physx_cuda
obs_mode=state, control_mode=pd_joint_delta_pos
render_mode=rgb_array, sensor_details=RGBD(320x180), RGBD(320x180), RGBD(320x180)
sim_freq=120, control_freq=60
observation space: Box(-inf, inf, (4, 18), float32)
(single) action space: Box(-1.0, 1.0, (8,), float32)
# -------------------------------------------------------------------------- #
start recording env.step metrics
env.step: 1129.465 steps/s, 282.366 parallel steps/s, 1000 steps in 3.542s
CPU mem: 1875.879 MB, GPU mem: 1232.527 MB
start recording pick_and_lift_env.step metrics
pick_and_lift_env.step: 873.967 steps/s, 218.492 parallel steps/s, 200 steps in 0.915s
CPU mem: 1879.816 MB, GPU mem: 1232.527 MB
start recording env.step+env.reset metrics
env.step+env.reset: 1072.118 steps/s, 268.030 parallel steps/s, 1000 steps in 3.731s
CPU mem: 1880.117 MB, GPU mem: 1232.527 MB
Running benchmark for env_id: CartpoleBalanceBenchmark-v1, num_envs: 4
2025-02-15 13:28:56,049 - mani_skill - WARNING - Currently ManiSkill does not support loading plane geometries from MJCFs
# -------------------------------------------------------------------------- #
Task ID: CartpoleBalanceBenchmark-v1, 4 parallel environments, sim_backend=physx_cuda
obs_mode=state, control_mode=pd_joint_delta_pos
render_mode=rgb_array, sensor_details=RGBD(320x180), RGBD(320x180), RGBD(320x180)
sim_freq=120, control_freq=60
observation space: Box(-inf, inf, (4, 10), float32)
(single) action space: Box(-1.0, 1.0, (1,), float32)
# -------------------------------------------------------------------------- #
start recording env.step metrics
env.step: 3204.033 steps/s, 801.008 parallel steps/s, 1000 steps in 1.248s
CPU mem: 2595.758 MB, GPU mem: 2322.527 MB
start recording env.step+env.reset metrics
env.step+env.reset: 3196.594 steps/s, 799.149 parallel steps/s, 1000 steps in 1.251s
CPU mem: 2597.621 MB, GPU mem: 2322.527 MB
Running benchmark for env_id: FrankaMoveBenchmark-v1, num_envs: 4
# -------------------------------------------------------------------------- #
Task ID: FrankaMoveBenchmark-v1, 4 parallel environments, sim_backend=physx_cuda
obs_mode=state, control_mode=pd_joint_delta_pos
render_mode=rgb_array, sensor_details=RGBD(320x180), RGBD(320x180), RGBD(320x180)
sim_freq=120, control_freq=60
observation space: Box(-inf, inf, (4, 18), float32)
(single) action space: Box(-1.0, 1.0, (8,), float32)
# -------------------------------------------------------------------------- #
start recording env.step metrics
env.step: 1542.942 steps/s, 385.736 parallel steps/s, 1000 steps in 2.592s
CPU mem: 3278.004 MB, GPU mem: 3410.527 MB
start recording env.step+env.reset metrics
env.step+env.reset: 1556.721 steps/s, 389.180 parallel steps/s, 1000 steps in 2.570s
CPU mem: 3279.551 MB, GPU mem: 3410.527 MB
"""

# 定义正则表达式模式
pattern = r"Running benchmark for env_id: (\w+), num_envs: (\d+).*?start recording env.step metrics.*?env.step: (\d+\.\d+) steps/s, (\d+\.\d+) parallel steps/s, (\d+) steps in (\d+\.\d+)s.*?CPU mem: (\d+\.\d+) MB, GPU mem: (\d+\.\d+) MB.*?start recording pick_and_lift_env.step metrics.*?pick_and_lift_env.step: (\d+\.\d+) steps/s, (\d+\.\d+) parallel steps/s, (\d+) steps in (\d+\.\d+)s.*?CPU mem: (\d+\.\d+) MB, GPU mem: (\d+\.\d+) MB.*?start recording env.step\+env.reset metrics.*?env.step\+env.reset: (\d+\.\d+) steps/s, (\d+\.\d+) parallel steps/s, (\d+) steps in (\d+\.\d+)s.*?CPU mem: (\d+\.\d+) MB, GPU mem: (\d+\.\d+) MB"

# 查找所有匹配项
matches = re.findall(pattern, result_text, re.DOTALL)

# 提取数据并存储到列表中
data = []
for match in matches:
    env_id = match[0]
    num_envs = int(match[1])
    env_step_steps_per_second = float(match[2])
    env_step_parallel_steps_per_second = float(match[3])
    env_step_total_steps = int(match[4])
    env_step_execution_time = float(match[5])
    env_step_cpu_mem = float(match[6])
    env_step_gpu_mem = float(match[7])
    pick_and_lift_steps_per_second = float(match[8])
    pick_and_lift_parallel_steps_per_second = float(match[9])
    pick_and_lift_total_steps = int(match[10])
    pick_and_lift_execution_time = float(match[11])
    pick_and_lift_cpu_mem = float(match[12])
    pick_and_lift_gpu_mem = float(match[13])
    env_step_reset_steps_per_second = float(match[14])
    env_step_reset_parallel_steps_per_second = float(match[15])
    env_step_reset_total_steps = int(match[16])
    env_step_reset_execution_time = float(match[17])
    env_step_reset_cpu_mem = float(match[18])
    env_step_reset_gpu_mem = float(match[19])

    data.append({
        "Env ID": env_id,
        "Num Envs": num_envs,
        "Env Step Steps/s": env_step_steps_per_second,
        "Env Step Parallel Steps/s": env_step_parallel_steps_per_second,
        "Env Step Total Steps": env_step_total_steps,
        "Env Step Execution Time (s)": env_step_execution_time,
        "Env Step CPU Mem (MB)": env_step_cpu_mem,
        "Env Step GPU Mem (MB)": env_step_gpu_mem,
        "Pick and Lift Steps/s": pick_and_lift_steps_per_second,
        "Pick and Lift Parallel Steps/s": pick_and_lift_parallel_steps_per_second,
        "Pick and Lift Total Steps": pick_and_lift_total_steps,
        "Pick and Lift Execution Time (s)": pick_and_lift_execution_time,
        "Pick and Lift CPU Mem (MB)": pick_and_lift_cpu_mem,
        "Pick and Lift GPU Mem (MB)": pick_and_lift_gpu_mem,
        "Env Step + Reset Steps/s": env_step_reset_steps_per_second,
        "Env Step + Reset Parallel Steps/s": env_step_reset_parallel_steps_per_second,
        "Env Step + Reset Total Steps": env_step_reset_total_steps,
        "Env Step + Reset Execution Time (s)": env_step_reset_execution_time,
        "Env Step + Reset CPU Mem (MB)": env_step_reset_cpu_mem,
        "Env Step + Reset GPU Mem (MB)": env_step_reset_gpu_mem
    })

# 创建 DataFrame
df = pd.DataFrame(data)

# 显示 DataFrame
print(df)