import gymnasium as gym
import time
import mani_skill
import pandas as pd

def create_env(env_id, num_envs, obs_mode, control_mode, shader_pack, camera_width, camera_height):
    return gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        human_render_camera_configs=dict(shader_pack=shader_pack, width=camera_width, height=camera_height),
        sensor_configs=dict(shader_pack=shader_pack),
        render_mode="human",
    )

def measure_render_speed(env, num_frames=100):
    start_time = time.time()
    for _ in range(num_frames):
        env.get_obs()
    end_time = time.time()
    return num_frames / (end_time - start_time)

def main():
    # Read parameters from CSV files
    isaac_lab_df = pd.read_csv('/home/zcm/lightrobot/ManiSkill/docs/source/user_guide/additional_resources/benchmarking_results/rtx_4090/isaac_lab.csv')
    maniskill_df = pd.read_csv('/home/zcm/lightrobot/ManiSkill/docs/source/user_guide/additional_resources/benchmarking_results/rtx_4090/maniskill.csv')

    # Combine unique parameters from both dataframes
    combined_df = pd.concat([isaac_lab_df, maniskill_df]).drop_duplicates(subset=['env_id', 'obs_mode', 'num_envs', 'num_cameras', 'camera_width', 'camera_height'])

    shader_packs = ["default", "rt", "rt-med", "rt-fast", "minimal"]
    
    results = []

    print("Observation rendering speed comparison:")
    print("-" * 60)
    for _, row in combined_df.iterrows():
        env_id = row['env_id']
        num_envs = row['num_envs']
        obs_mode = row['obs_mode']
        control_mode = row.get('control_mode', 'pd_joint_delta_pos')  # Default to 'pd_joint_delta_pos' if not present
        camera_width = row['camera_width']
        camera_height = row['camera_height']

        for shader in shader_packs:
            try:
                env = create_env(env_id, num_envs, obs_mode, control_mode, shader, camera_width, camera_height)
                obs, _ = env.reset(seed=1)
                fps = measure_render_speed(env)
                print(f"{env_id} - {obs_mode} - {num_envs} envs - {shader:10} rendering: {fps:.2f} FPS")
                results.append({
                    'env_id': env_id,
                    'num_envs': num_envs,
                    'obs_mode': obs_mode,
                    'control_mode': control_mode,
                    'shader_pack': shader,
                    'camera_width': camera_width,
                    'camera_height': camera_height,
                    'fps': fps
                })
                env.close()
            except Exception as e:
                print(f"Error with {env_id} - {obs_mode} - {num_envs} envs - {shader} shader: {str(e)}")
    print("-" * 60)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('render_speed_results.csv', index=False)
    print("Results saved to 'render_speed_results.csv'")

if __name__ == "__main__":
    main()
