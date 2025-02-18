#!/bin/bash

# Source the conda environment
source /home/zcm/miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# Source the Isaac Sim Python environment setup
source /home/zcm/.local/share/ov/pkg/isaac-sim-comp-check-4.2.0/setup_python_env.sh

# Modify PYTHONPATH to include the conda environment's site-packages
export PYTHONPATH="/home/zcm/miniconda3/envs/maniskill/lib/python3.12/site-packages:$PYTHONPATH"

# Print debugging information
echo "Python executable: $(which python)" > debug_log.txt
echo "Python version:" >> debug_log.txt
python --version >> debug_log.txt 2>&1
echo "Python path:" >> debug_log.txt
python -c "import sys; print('\n'.join(sys.path))" >> debug_log.txt
echo "PYTHONPATH:" >> debug_log.txt
echo $PYTHONPATH >> debug_log.txt
echo "Attempting to import omni.isaac:" >> debug_log.txt
python -c "import omni.isaac; print('Successfully imported omni.isaac')" >> debug_log.txt 2>&1
echo "Attempting to import numpy:" >> debug_log.txt
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); print(f'NumPy location: {numpy.__file__}')" >> debug_log.txt 2>&1
echo "List of all 'omni' modules:" >> debug_log.txt
python -c "import pkgutil; import omni; print('\n'.join(m.name for m in pkgutil.iter_modules(omni.__path__, omni.__name__ + '.')))" >> debug_log.txt 2>&1
echo "Detailed sys.path:" >> debug_log.txt
python -c "import sys; [print(f'{p}:\n  ' + '\n  '.join(sorted(set(__import__('os').listdir(p))))) for p in sys.path if __import__('os').path.isdir(p)]" >> debug_log.txt 2>&1

# Change to the directory containing the script
cd "$(dirname "$0")"

# Run the benchmarking script and redirect output to a log file
python isaac_lab_gpu_sim.py "$@" > benchmark_log.txt 2>&1

echo "Benchmark completed. Check benchmark_log.txt and debug_log.txt for details."
