import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
from pynput import keyboard
import os
import torch
from torchvision.utils import save_image
save_dir = "/home/zcm/Pictures"  # 定义图像保存目录
# Initialize a lock for thread synchronization
lock = threading.Lock()

# Robot joint angle limits (adjust according to your robot configuration)
control_qlimit = np.array([
    [-2.1, -3.1, -0.0, -1.375, -1.57],
    [ 2.1,  0.0,  3.1,  1.475,  1.57]
])

# Initial joint angles (global variable)
initial_theta = np.array([0.0, -3.14, 3.14, 0.0, -1.57])
current_theta = initial_theta.copy()

# Define rotation matrices (return 4x4 homogeneous transformation matrices)
def R_x(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [1,    0,     0, 0],
        [0,  cos, -sin, 0],
        [0,  sin,  cos, 0],
        [0,    0,     0, 1]
    ])

def R_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [ cos, 0, sin, 0],
        [   0, 1,   0, 0],
        [-sin, 0, cos, 0],
        [   0, 0,   0, 1]
    ])

def R_z(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [cos, -sin, 0, 0],
        [sin,  cos, 0, 0],
        [  0,    0, 1, 0],
        [  0,    0, 0, 1]
    ])

# Define translation matrices (4x4 homogeneous transformation matrices)
def T_x(a):
    return np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def T_y(a):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, a],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def T_z(a):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, a],
        [0, 0, 0, 1]
    ])

# Forward kinematics function
def FK(theta):

    theta1, theta2, theta3, theta4, theta5 = theta

    # Joint 1
    E1 = T_x(0.0612)
    E2 = T_z(0.0598)
    E3 = R_z(theta1)

    # Joint 2
    E4 = T_x(0.02943)
    E5 = T_z(0.05504)
    E6 = R_y(theta2)

    # Joint 3
    E7 = T_x(0.1127)
    E8 = T_z(-0.02798)
    E9 = R_y(theta3)

    # Joint 4
    E10 = T_x(0.13504)
    E11 = T_z(0.00519)
    E12 = R_y(theta4)

    # Joint 5
    E13 = T_x(0.0593)
    E14 = T_z(0.00996)
    E15 = R_x(theta5)

    
    T_end = E1 @ E2 @ E3 @ E4 @ E5 @ E6 @ E7 @ E8 @ E9 @ E10 @ E11 @ E12 @ E13 @ E14 @ E15

    
    location = T_end[0:3, 3]
    orientation = T_end[0:3, 0:3]

    return T_end, location, orientation

# Calculate the Jacobian matrix
def Jacobian(theta):

    theta1, theta2, theta3, theta4, theta5 = theta

    # Joint 1
    E1 = T_x(0.0612)
    E2 = T_z(0.0598)
    E3 = R_z(theta1)

    # Joint 2
    E4 = T_x(0.02943)
    E5 = T_z(0.05504)
    E6 = R_y(theta2)

    # Joint 3
    E7 = T_x(0.1127)
    E8 = T_z(-0.02798)
    E9 = R_y(theta3)

    # Joint 4
    E10 = T_x(0.13504)
    E11 = T_z(0.00519)
    E12 = R_y(theta4)

    # Joint 5
    E13 = T_x(0.0593)
    E14 = T_z(0.00996)
    E15 = R_x(theta5)

    # Gripper
    E16 = T_x(0.09538)

    # Compute transformation matrices
    T0 = np.eye(4)
    # Position of base
    p0 = T0[0:3, 3]
    
    # Joint 1 transform up to before rotation
    T1 = T0 @ E1 @ E2
    p1 = T1[0:3, 3]
    z1 = np.array([0, 0, 1])  # Rotation axis of joint 1 in base frame

    # Joint 1 rotation
    T1_rot = T1 @ E3

    # Joint 2
    T2 = T1_rot @ E4 @ E5
    p2 = T2[0:3, 3]
    z2 = T1_rot @ np.array([[0], [1], [0], [0]])  # Rotation axis of joint 2 in base frame
    z2 = z2[0:3, 0]

    T2_rot = T2 @ E6

    # Joint 3
    T3 = T2_rot @ E7 @ E8
    p3 = T3[0:3, 3]
    z3 = T2_rot @ np.array([[0], [1], [0], [0]])
    z3 = z3[0:3, 0]
    
    T3_rot = T3 @ E9

    # Joint 4
    T4 = T3_rot @ E10 @ E11
    p4 = T4[0:3, 3]
    z4 = T3_rot @ np.array([[0], [1], [0], [0]])
    z4 = z4[0:3, 0]
    
    T4_rot = T4 @ E12

    # Joint 5
    T5 = T4_rot @ E13 @ E14
    p5 = T5[0:3, 3]
    z5 = T4_rot @ np.array([[1], [0], [0], [0]])
    z5 = z5[0:3, 0]
    
    T5_rot = T5 @ E15

    # Gripper
    T_end = T5_rot @ E16
    p_end = T_end[0:3, 3]

    # Initialize Jacobian
    J = np.zeros((6, 5))

    # Joint 1
    J[:, 0] = np.concatenate((np.cross(z1, p_end - p1), z1))

    # Joint 2
    J[:, 1] = np.concatenate((np.cross(z2, p_end - p2), z2))

    # Joint 3
    J[:, 2] = np.concatenate((np.cross(z3, p_end - p3), z3))

    # Joint 4
    J[:, 3] = np.concatenate((np.cross(z4, p_end - p4), z4))

    # Joint 5
    J[:, 4] = np.concatenate((np.cross(z5, p_end - p5), z5))

    return J

# Inverse kinematics function
def IK(desired_T, initial_theta, max_iterations=1000, tolerance=1e-6):
    theta = np.array(initial_theta, dtype=float)
    lambda_identity = 0.01  # Damping factor

    # Error weights
    w_position = 1.0 
    w_orientation = 0.1 

    for i in range(max_iterations):
        current_T, _, _ = FK(theta)

        pos_error = desired_T[0:3, 3] - current_T[0:3, 3]

        R_current = current_T[0:3, 0:3]
        R_desired = desired_T[0:3, 0:3]
        R_error = R_desired @ R_current.T

        trace = np.trace(R_error)
        # Prevent numerical errors from causing the arccos input to exceed the range
        trace = np.clip(trace, -1.0, 3.0)
        angle = np.arccos((trace - 1) / 2)
        if np.isnan(angle):
            angle = 0.0
        if angle < 1e-6:
            rot_error_vec = np.zeros(3)
        else:
            rx = R_error[2, 1] - R_error[1, 2]
            ry = R_error[0, 2] - R_error[2, 0]
            rz = R_error[1, 0] - R_error[0, 1]
            rot_axis = np.array([rx, ry, rz])
            rot_axis_norm = np.linalg.norm(rot_axis)
            if rot_axis_norm < 1e-6:
                rot_error_vec = np.zeros(3)
            else:
                rot_axis = rot_axis / rot_axis_norm
                rot_error_vec = rot_axis * angle

        # Combine position and rotation errors
        error = np.concatenate((pos_error * w_position, rot_error_vec * w_orientation))

        if np.linalg.norm(error) < tolerance:
            print(f"Converged at iteration {i+1}.")
            return theta

        # Calculate the Jacobian matrix
        J = Jacobian(theta)

        # Weighted Jacobian matrix and error
        W = np.diag([w_position]*3 + [w_orientation]*3)
        J_weighted = W @ J  # 6x5
        error_weighted = W @ error  # 6x1

        try:
            JJT = J_weighted @ J_weighted.T  # 6x6
            inv_JJT = np.linalg.inv(JJT + lambda_identity * np.eye(6))  # 6x6
            J_pinv = J_weighted.T @ inv_JJT  # 5x6
        except np.linalg.LinAlgError:
            print("Non-invertible Jacobian matrix, possible singularity encountered.")
            return theta

        # Calculate the joint angle update
        delta_theta = J_pinv @ error_weighted  # 5x1

        # Update the joint angles
        theta += delta_theta

        # Apply joint angle limits
        theta = np.clip(theta, control_qlimit[0], control_qlimit[1])

    print("Reached the maximum number of iterations, did not fully converge.")
    return theta
def IK1(desired_T, initial_theta, max_iterations=1000, tolerance=1e-6):
    """
    Calculate the inverse kinematics.
    :param desired_T: The desired homogeneous transformation matrix of the end - effector.
    :param initial_theta: The initial joint angle list [theta1, theta2, theta3, theta4, theta5].
    :param max_iterations: The maximum number of iterations.
    :param tolerance: The error tolerance.
    :return: The joint angle list.
    """
    theta = np.array(initial_theta, dtype=float)
    lambda_identity = 0.01  # Damping factor

    for i in range(max_iterations):
        current_T, _, _ = FK(theta)
        # Position error
        pos_error = desired_T[0:3, 3] - current_T[0:3, 3]
        # Rotation error (using the difference of rotation matrices)
        R_current = current_T[0:3, 0:3]
        R_desired = desired_T[0:3, 0:3]
        R_error = R_desired @ R_current.T
        # Convert the rotation error to axis - angle form
        angle = np.arccos((np.trace(R_error) - 1) / 2)
        if angle < 1e-6:
            rot_error_vec = np.zeros(3)
        else:
            rx = R_error[2, 1] - R_error[1, 2]
            ry = R_error[0, 2] - R_error[2, 0]
            rz = R_error[1, 0] - R_error[0, 1]
            rot_axis = np.array([rx, ry, rz])
            rot_axis = rot_axis / (2 * np.sin(angle))
            rot_error_vec = rot_axis * angle
        # Combine position and rotation errors
        error = np.concatenate((pos_error, rot_error_vec))
        if np.linalg.norm(error) < tolerance:
            print(f"Converged at iteration {i + 1}.")
            return theta
        # Calculate the Jacobian matrix
        J = Jacobian(theta)
        # Calculate the pseudo - inverse J_pinv = J.T @ (J @ J.T + lambda * I)^-1
        try:
            JJT = J @ J.T  # 6x6
            inv_JJT = np.linalg.inv(JJT + lambda_identity * np.eye(6))  # 6x6
            J_pinv = J.T @ inv_JJT  # 5x6
        except np.linalg.LinAlgError:
            print("The Jacobian matrix cannot be inverted. A singularity may be encountered.")
            return theta
        # Calculate the joint angle update
        delta_theta = J_pinv @ error  # 5x6 * 6x1 = 5x1
        # Update the joint angles
        theta += delta_theta
        # Optional: Add joint angle limits
        # min_theta = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi])
        # max_theta = np.array([np.pi, np.pi/2, np.pi/2, np.pi/2, np.pi])
        # theta = np.clip(theta, min_theta, max_theta)
    print("Reached the maximum number of iterations without full convergence.")
    return theta
# Function to plot the robot arm
def plot_robot_arm(theta):
    T0 = np.eye(4)

    T1 = T0 @ T_x(0.0612) @ T_z(0.0598) @ R_z(theta[0])

    T2 = T1 @ T_x(0.02943) @ T_z(0.05504) @ R_y(theta[1])

    T3 = T2 @ T_x(0.1127) @ T_z(-0.02798) @ R_y(theta[2])

    T4 = T3 @ T_x(0.13504) @ T_z(0.00519) @ R_y(theta[3])

    T5 = T4 @ T_x(0.0593) @ T_z(0.00996) @ R_x(theta[4])

    T6 = T5 @ T_x(0.09538)  # Gripper

    # Extract the coordinates of each joint
    joints = [
        T0[0:3, 3],
        T1[0:3, 3],
        T2[0:3, 3],
        T3[0:3, 3],
        T4[0:3, 3],
        T5[0:3, 3],
        T6[0:3, 3]
    ]

    ax.cla()

    x = [joint[0] for joint in joints]
    y = [joint[1] for joint in joints]
    z = [joint[2] for joint in joints]

    ax.plot(x, y, z, '-o', label="Robot Arm", markersize=5)

    ax.set_title("3D Robot Arm Configuration")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.set_xlim([-0.2, 0.5])
    ax.set_ylim([-0.2, 0.5])
    ax.set_zlim([0, 0.6])

    ax.legend()
    plt.draw()
    plt.pause(0.001)

# Auxiliary function: Get Euler angles from a rotation matrix (ZYX order)
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# Auxiliary function: Generate a rotation matrix from Euler angles (ZYX order)
def eulerAnglesToRotationMatrix(theta):
    R = R_z(theta[2]) @ R_y(theta[1]) @ R_x(theta[0])
    return R

# Increment mapping for keyboard control
key_to_delta = {
    'w': ('x', -0.005),
    'x': ('x',  0.005),
    'a': ('y',  0.005),
    'd': ('y', -0.005),
    'r': ('z',  0.005),
    'f': ('z', -0.005),
    'z': ('roll',    np.deg2rad(1)),
    'c': ('roll',   -np.deg2rad(1)),
    't': ('pitch',   np.deg2rad(1)),
    'g': ('pitch',  -np.deg2rad(1)),
    'y': ('yaw',     np.deg2rad(1)),
    'h': ('yaw',    -np.deg2rad(1))
}

# Keyboard press event handler
def on_press_key(key):
    global desired_T, current_theta
    try:
        k = key.char.lower()
        if k in key_to_delta:
            with lock:
                param, delta = key_to_delta[k]
                if param in ['x', 'y', 'z']:
                    index = {'x': 0, 'y': 1, 'z': 2}[param]
                    desired_T[0:3, 3][index] += delta
                elif param in ['roll', 'pitch', 'yaw']:
                    R_current = desired_T[0:3, 0:3]
                    euler_angles = rotationMatrixToEulerAngles(R_current)
                    if param == 'roll':
                        euler_angles[0] += delta
                    elif param == 'pitch':
                        euler_angles[1] += delta
                    elif param == 'yaw':
                        euler_angles[2] += delta
                    desired_T[0:3, 0:3] = eulerAnglesToRotationMatrix(euler_angles)
            print(f'Key pressed: {k}')
    except AttributeError:
        pass  # Handle special keys

def on_release_key(key):
    pass  # Add logic for key release if needed

# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press_key, on_release=on_release_key)
listener.start()
env = gym.make(
    "PickCubeSO100-vlm", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
    sensor_configs=dict(shader_pack="rt"),  # 为所有传感器相机设置"rt"着色器包
    render_mode="human",
    cube_position=[-0.45, 0.2],
    cube_rotation=[1, 0, 0, 0]
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)
obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
current_T, _, _ = FK(initial_theta)
action = np.zeros(6)
np.random.seed(1)
if __name__ == "__main__":
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize desired_T to the current end-effector pose
    current_T, location, orientation = FK(initial_theta)
    
    desired_T = current_T.copy()
    # plot_robot_arm(initial_theta)
    while True:
        with lock:
            # Calculate inverse kinematics
            solution_theta = IK(desired_T, current_theta)
            if solution_theta is not None:
                # Update the joint angles
                current_theta = solution_theta
                # Plot the robot
                # plot_robot_arm(current_theta)
            else:
                print("Unable to solve the inverse kinematics problem.")
           
            if solution_theta is not None:
                # 构造完整的动作向量：7个关节角度 + 1个夹持器控制值
                
                action[:5] =solution_theta   # 前7个值为关节角度
                action[5] = 0.5  # 第8个值为夹持器控制
                action = np.array(action)
                # print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()  # a display is required to render
            print(obs)
            rgb_image = obs['sensor_data']['base_camera']['rgb']
            # 由于图像数据通常为 (C, H, W, 3) 格式（C 为图像数量，这里通常为 1），如果 C 为 1 则去掉该维度
            if rgb_image.ndim == 4 and rgb_image.shape[0] == 1:
                rgb_image = rgb_image.squeeze(0)
            # 将数据类型转换为 float 并归一化到 [0, 1] 范围
            rgb_image = rgb_image.float() / 255.0
            # 调整通道顺序，从 (H, W, C) 到 (C, H, W)
            rgb_image = rgb_image.permute(2, 0, 1)

            # 保存 RGB 图像
            rgb_image_path = os.path.join(save_dir, 'rgb_image.png')
            save_image(rgb_image, rgb_image_path)

            # 提取深度图像
            depth_image = obs['sensor_data']['base_camera']['depth']
            # 同样，如果有多余维度（如图像数量为 1）则去掉
            if depth_image.ndim == 4 and depth_image.shape[0] == 1:
                depth_image = depth_image.squeeze(0)
            # 去掉深度图像的单通道维度
            depth_image = depth_image.squeeze(-1)
            # 将数据类型转换为 float 并归一化到 [0, 1] 范围（假设深度值有一定范围）
            depth_min = depth_image.min()
            depth_max = depth_image.max()
            depth_image = (depth_image - depth_min) / (depth_max - depth_min + 1e-8)
            depth_image = depth_image.float()

            # 保存深度图像，去掉 cmap 参数
            depth_image_path = os.path.join(save_dir, 'depth_image.png')
            save_image(depth_image, depth_image_path)

            print(f"RGB 图像已保存到 {rgb_image_path}")
            print(f"深度图像已保存到 {depth_image_path}")
        # Add an appropriate delay to avoid excessive CPU usage
 
