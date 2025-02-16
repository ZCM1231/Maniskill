import gymnasium as gym
import mani_skill.envs
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
def compute_inverse_kinematics(self, p_target: np.ndarray,
                                        quat_target: np.ndarray,
                                        max_iterations: int=10000,
                                        lr: float=0.1,
                                        tolerance: float=1e-5):
        '''
        This is an iterative method. It is very slow for now
        '''
        start_time = time.time()
        # initialize
        q, _, _, _  = self.get_current_joint_states()
        for i in range(max_iterations):
            # Compute current position and orientation (this is a placeholder function)
            p, quat = self.compute_forward_kinematics(q)
            p_error = p_target - p
            rot_error = self.compute_quat_error2(quat, quat_target)
            error = np.concatenate((p_error, rot_error))
            # Check for convergence
            if np.linalg.norm(error) < tolerance:
                print('Found IK solution within tolerance')
                break
            J = self.compute_Jacobian(q)
            q += lr * np.dot(J.T, error)
        exec_time = time.time() - start_time
        print(f'exec time = {exec_time} seconds.')
        return q

def compute_quat_error2(self, quat_target: np.ndarray, quat_current: np.ndarray):
        '''
        This works, but weird. Why though?
        '''
        quat_current = np.array(quat_current)
        if np.dot(quat_target, quat_current) < 0:
            quat_current = -1.0 * quat_current
        quat_current = R.from_quat(quat_current)
        quat_target = R.from_quat(quat_target)
        error_quat = quat_current.inv() * quat_target
        error_quat = error_quat.as_quat()
        error = -quat_current.as_matrix() @ error_quat[:3]
        return error

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # a display is required to render
env.close()