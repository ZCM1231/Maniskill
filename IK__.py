import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 机器人关节角度限制（根据URDF）
control_qlimit = np.array([
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
])

# 初始关节角度
initial_theta = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
current_theta = initial_theta.copy()

# 从URDF中提取的DH参数
dh_params = [
    {'a': 0,       'd': 0.333,  'alpha': 0},           # Joint 1
    {'a': 0,       'd': 0,      'alpha': -np.pi/2},    # Joint 2
    {'a': 0,       'd': 0.316,  'alpha': np.pi/2},     # Joint 3
    {'a': 0.0825,  'd': 0,      'alpha': np.pi/2},     # Joint 4
    {'a': -0.0825, 'd': 0.384,  'alpha': -np.pi/2},    # Joint 5
    {'a': 0,       'd': 0,      'alpha': np.pi/2},     # Joint 6
    {'a': 0.088,   'd': 0.107,  'alpha': np.pi/2}      # Joint 7
]

# 末端执行器偏移（根据URDF中的TCP定义）
tcp_offset = np.array([0, 0, 0.1034])

# 定义DH变换矩阵
def T_matrix(a, d, alpha, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

# 正向运动学函数
def FK(theta):
    T = np.eye(4)
    for i in range(7):
        T_i = T_matrix(dh_params[i]['a'], dh_params[i]['d'], dh_params[i]['alpha'], theta[i])
        T = T @ T_i
    # 添加末端执行器偏移
    T_end = T.copy()
    T_end[0:3, 3] += T[0:3, 0:3] @ tcp_offset
    return T_end, T[0:3, 3], T[0:3, 0:3]  # 返回末端位姿、位置和方向

# 计算雅可比矩阵
def Jacobian(theta):
    J = np.zeros((6, 7))
    T = np.eye(4)
    z_axes = []
    positions = [T[0:3, 3]]

    # 前向计算各关节的位置和轴向
    for i in range(7):
        T_i = T_matrix(dh_params[i]['a'], dh_params[i]['d'], dh_params[i]['alpha'], theta[i])
        T = T @ T_i
        z_axes.append(T[0:3, 2])
        positions.append(T[0:3, 3])

    # 末端位置（包括TCP偏移）
    T_end = T.copy()
    T_end[0:3, 3] += T[0:3, 0:3] @ tcp_offset
    p_end = T_end[0:3, 3]

    for i in range(7):
        z = z_axes[i]
        p = positions[i]
        J[0:3, i] = np.cross(z, p_end - p)
        J[3:6, i] = z

    return J

# 逆运动学函数
def IK(desired_T, initial_theta, max_iterations=1000, tolerance=1e-6):
    theta = np.array(initial_theta, dtype=float)
    lambda_identity = 0.01  # 阻尼系数

    for i in range(max_iterations):
        current_T, _, _ = FK(theta)
        pos_error = desired_T[0:3, 3] - current_T[0:3, 3]

        R_current = current_T[0:3, 0:3]
        R_desired = desired_T[0:3, 0:3]
        R_error = R_desired @ R_current.T

        # 旋转误差转换为轴角表示
        angle = np.arccos((np.trace(R_error) - 1) / 2)
        if np.isnan(angle) or angle < 1e-6:
            rot_error_vec = np.zeros(3)
        else:
            rot_error_vec = (1/(2*np.sin(angle))) * np.array([
                R_error[2,1] - R_error[1,2],
                R_error[0,2] - R_error[2,0],
                R_error[1,0] - R_error[0,1]
            ]) * angle

        # 综合误差
        error = np.concatenate((pos_error, rot_error_vec))

        if np.linalg.norm(error) < tolerance:
            print(f"Converged at iteration {i+1}.")
            return theta

        # 计算雅可比矩阵
        J = Jacobian(theta)

        # 阻尼最小二乘法
        try:
            JTJ = J.T @ J + lambda_identity * np.eye(7)
            delta_theta = np.linalg.solve(JTJ, J.T @ error)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered during IK computation.")
            return theta

        # 更新关节角度
        theta += delta_theta

        # 应用关节限位
        theta = np.clip(theta, control_qlimit[0], control_qlimit[1])

    print("Reached the maximum number of iterations, did not fully converge.")
    return theta

def plot_robot(theta):
    """绘制机器人当前构型"""
    ax.cla()
    T = np.eye(4)
    x_points = [0]
    y_points = [0]
    z_points = [0]

    for i in range(7):
        T_i = T_matrix(dh_params[i]['a'], dh_params[i]['d'], dh_params[i]['alpha'], theta[i])
        T = T @ T_i
        x_points.append(T[0,3])
        y_points.append(T[1,3])
        z_points.append(T[2,3])

    # 添加末端执行器位置
    T_end = T.copy()
    T_end[0:3, 3] += T[0:3, 0:3] @ tcp_offset
    x_points.append(T_end[0,3])
    y_points.append(T_end[1,3])
    z_points.append(T_end[2,3])

    ax.plot(x_points, y_points, z_points, '-o', markersize=5)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.draw()
    plt.pause(0.001)

if __name__ == "__main__":
    # 创建图形窗口
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 计算初始位姿
    current_T, _, _ = FK(initial_theta)
    
    # 设置目标位姿（这里可以根据需要修改）
    desired_T = current_T.copy()
    desired_T[0:3, 3] += np.array([0.1, 0.1, 0.1])  # 移动末端执行器

    while True:
        # 计算逆运动学
        solution_theta = IK(desired_T, current_theta)
        if solution_theta is not None:
            # 更新关节角度
            current_theta = solution_theta
            # 绘制机器人
            plot_robot(current_theta)
        else:
            print("Unable to solve the inverse kinematics problem.")
        plt.pause(0.05)
