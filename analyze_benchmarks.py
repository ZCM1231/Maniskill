import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('benchmark_results.csv')

# Set up the plot style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Benchmark Results Analysis', fontsize=16)

# Plot 1: env.step performance
axs[0, 0].set_title('env.step Performance')
for env in df['Environment ID'].unique():
    env_data = df[df['Environment ID'] == env]
    axs[0, 0].plot(env_data['Num Parallel Envs'], env_data['env.step (steps/s)'], marker='o', label=env)
axs[0, 0].set_xlabel('Number of Parallel Environments')
axs[0, 0].set_ylabel('Steps per Second')
axs[0, 0].set_xscale('log', base=2)
axs[0, 0].set_yscale('log')
axs[0, 0].legend()

# Plot 2: env.step+env.reset performance
axs[0, 1].set_title('env.step+env.reset Performance')
for env in df['Environment ID'].unique():
    env_data = df[df['Environment ID'] == env]
    axs[0, 1].plot(env_data['Num Parallel Envs'], env_data['env.step+env.reset (steps/s)'], marker='o', label=env)
axs[0, 1].set_xlabel('Number of Parallel Environments')
axs[0, 1].set_ylabel('Steps per Second')
axs[0, 1].set_xscale('log', base=2)
axs[0, 1].set_yscale('log')
axs[0, 1].legend()

# Plot 3: CPU Memory Usage
axs[1, 0].set_title('CPU Memory Usage')
for env in df['Environment ID'].unique():
    env_data = df[df['Environment ID'] == env]
    axs[1, 0].plot(env_data['Num Parallel Envs'], env_data['CPU Memory (MB)'], marker='o', label=env)
axs[1, 0].set_xlabel('Number of Parallel Environments')
axs[1, 0].set_ylabel('CPU Memory (MB)')
axs[1, 0].set_xscale('log', base=2)
axs[1, 0].legend()

# Plot 4: GPU Memory Usage
axs[1, 1].set_title('GPU Memory Usage')
for env in df['Environment ID'].unique():
    env_data = df[df['Environment ID'] == env]
    axs[1, 1].plot(env_data['Num Parallel Envs'], env_data['GPU Memory (MB)'], marker='o', label=env)
axs[1, 1].set_xlabel('Number of Parallel Environments')
axs[1, 1].set_ylabel('GPU Memory (MB)')
axs[1, 1].set_xscale('log', base=2)
axs[1, 1].legend()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('benchmark_analysis.png', dpi=300, bbox_inches='tight')
print("Benchmark analysis plot saved as 'benchmark_analysis.png'")
