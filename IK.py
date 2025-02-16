import numpy as np
from scipy.spatial.transform import Rotation as R


class PandaIK:
    def __init__(self):
        # DH parameters for Panda robot arm
        self.dh_params = {
            'd' : [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107],
            'a': [0, 0, 0, 0.0825, -0.0825, 0, 0.088],
            'alpha': [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2],
            'offset': [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
        }

        # Joint limits for Panda arm in radians
        self.joint_limits = {
            'lower': np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            'upper': np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        }

    def transform_matrix(self, theta, d, a, alpha):
        """Calculate individual transformation matrix for a single joint."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, q):
        """Compute forward kinematics for the given joint angles."""
        T = np.eye(4)
        for i in range(7):
            theta = q[i] + self.dh_params['offset'][i]
            d = self.dh_params['d'][i]
            a = self.dh_params['a'][i]
            alpha = self.dh_params['alpha'][i]
            
            Ti = self.transform_matrix(theta, d, a, alpha)
            T = T @ Ti
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation

    def compute_jacobian(self, q):
        """Compute Jacobian matrix using finite difference."""
        epsilon = 1e-6
        jacobian = np.zeros((6, 7))
        
        for i in range(7):
            q_epsilon = q.copy()
            q_epsilon[i] += epsilon
            
            pos1, rot1 = self.forward_kinematics(q)
            pos2, rot2 = self.forward_kinematics(q_epsilon)
            
            # Position Jacobian
            jacobian[:3, i] = (pos2 - pos1) / epsilon
            
            # Orientation Jacobian
            rot_diff = rot2 @ rot1.T
            angle = np.arccos((np.trace(rot_diff) - 1) / 2)
            if angle < 1e-6:
                jacobian[3:, i] = 0
            else:
                axis = np.array([
                    rot_diff[2, 1] - rot_diff[1, 2],
                    rot_diff[0, 2] - rot_diff[2, 0],
                    rot_diff[1, 0] - rot_diff[0, 1]
                ]) / (2 * np.sin(angle))
                jacobian[3:, i] = (axis * angle) / epsilon
                
        return jacobian

    def compute_orientation_error(self, current_rot_matrix, target_rot_matrix):
        """Compute orientation error using axis-angle representation."""
        error_rot_matrix = target_rot_matrix @ current_rot_matrix.T
        angle = np.arccos((np.trace(error_rot_matrix) - 1) / 2)
        
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            error_rot_matrix[2, 1] - error_rot_matrix[1, 2],
            error_rot_matrix[0, 2] - error_rot_matrix[2, 0],
            error_rot_matrix[1, 0] - error_rot_matrix[0, 1]
        ]) / (2 * np.sin(angle))
        
        return axis * angle

    def inverse_kinematics(self, target_pos, target_rot, max_iter=200, pos_tol=1e-3, ori_tol=0.1):
        """Solve inverse kinematics using DLS method."""
        q_init_guess = [
            np.zeros(7),  # Zero position
            np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]),  # Typical configuration
            np.random.uniform(self.joint_limits['lower'], self.joint_limits['upper'])  # Random configuration
        ]
        
        best_q = None
        min_error = float('inf')
        success = False

        for q_start in q_init_guess:
            q = q_start.copy()
            lambda_ = 0.01  # Initial damping factor
            last_error_norm = float('inf')
            
            for iter in range(max_iter):
                pos, rot = self.forward_kinematics(q)
                
                # Position error
                pos_error = target_pos - pos
                pos_error_norm = np.linalg.norm(pos_error)

                # Orientation error
                ori_error = self.compute_orientation_error(rot, target_rot)
                ori_error_norm = np.linalg.norm(ori_error)

                # Combined error
                error = np.concatenate([pos_error, ori_error])
                error_norm = np.linalg.norm(error)

                # Check convergence
                if pos_error_norm < pos_tol and ori_error_norm < ori_tol:
                    success = True
                    if error_norm < min_error:
                        min_error = error_norm
                        best_q = q.copy()
                    break

                # Adjust damping factor
                if error_norm < last_error_norm:
                    lambda_ = max(0.01, lambda_ * 0.5)
                else:
                    lambda_ = min(1.0, lambda_ * 2.0)
                
                last_error_norm = error_norm

                # Compute Jacobian
                J = self.compute_jacobian(q)
                
                # Update joint angles using DLS
                JJT = J @ J.T
                lambda_I = lambda_ * np.eye(6)
                try:
                    delta_q = J.T @ np.linalg.solve(JJT + lambda_I, error)
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered, trying different initial position")
                    break

                # Step size limit
                step_size = min(0.1, np.linalg.norm(delta_q))
                if np.linalg.norm(delta_q) > step_size:
                    delta_q = delta_q * step_size / np.linalg.norm(delta_q)
                
                # Update joint angles
                q_new = q + delta_q
                
                # Handle joint limits
                q_new = np.clip(q_new, self.joint_limits['lower'], self.joint_limits['upper'])
                
                # Check for convergence by step size
                if np.linalg.norm(q_new - q) < 1e-6:
                    if error_norm < min_error:
                        min_error = error_norm
                        best_q = q_new.copy()
                    break
                
                q = q_new

        if success:
            print(f"IK converged with error: {min_error}")
            return best_q, True
        else:
            print(f"Best solution found with error: {min_error}")
            return best_q if best_q is not None else q, False


# Example Usage
if __name__ == "__main__":
    panda = PandaIK()
    
    # Test a reachable target pose
    target_position = np.array([0.0, 0.0, 0.25])  # In workspace
    target_rotation = R.from_euler('xyz', [0, np.pi/2, 0]).as_matrix()  # Arbitrary orientation
    
    # Run inverse kinematics solver
    q_sol, success = panda.inverse_kinematics(
        target_position,
        target_rotation,
        pos_tol=1e-3,    # Position tolerance in meters
        ori_tol=0.1      # Orientation tolerance in radians ~5.7 degrees
    )
    
    if success:
        print("Found solution:", q_sol)
        # Verify results
        final_pos, final_rot = panda.forward_kinematics(q_sol)
        print("Final position:", final_pos)
        print("Position error:", np.linalg.norm(final_pos - target_position))
        
        # Display orientation error as axis-angle
        ori_error = R.from_matrix(final_rot @ target_rotation.T).as_rotvec()
        print("Orientation error (radians):", np.linalg.norm(ori_error))
    else:
        print("Could not find exact solution, best attempt:")
        print(q_sol)
