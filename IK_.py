
#pip install roboticstoolbox-python==1.0.3  # 确认版本兼容性
import math
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

def create_panda_chain():
    """创建 Franka Panda 的 DH 模型"""
    a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
    d = [0.333, 0, 0.316, 0.0, 0.384, 0.0, 0.107]
    alpha = [0.0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
    joints = [RevoluteDH(d=d[i], a=a[i], alpha=alpha[i]) for i in range(7)]
    robot = DHRobot(joints, name='Panda')
    return robot

def compute_inverse_kinematics(robot, target_pose):
    """逆运动学求解"""
    # 构造目标位姿
    target_frame = SE3(target_pose[:3]) * SE3.RPY(target_pose[3:], order='xyz')
    
    # 使用正确的参数名 T
    solution = robot.ikine_LM(
        T=target_frame,
        # 避免用零初始值尝试！
        q0 = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
        mask=[1, 1, 1, 1, 1, 1],
        tol=1e-1,
        ilimit=10000
    )
    
    if solution.success:
        return solution.q
    else:
        print("求解失败，请检查目标位姿是否可达")
        return None

if __name__ == "__main__":
    # 创建机器人模型
    panda = create_panda_chain()
    
    # 目标位姿 [x, y, z, roll, pitch, yaw]（单位：米和弧度）
    target_pose = [0.4, 0.4, 0.8, 0.0, 0.0, 0.0]
    
    # 计算逆运动学
    q_result = compute_inverse_kinematics(panda, target_pose)
    
    if q_result is not None:
        print("\n逆运动学解（弧度）:", q_result)
        print("验证正运动学结果:", panda.fkine(q_result))
