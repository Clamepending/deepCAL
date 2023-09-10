#!/bin/bash
# Job name:
#SBATCH --job-name=hello_world
#
# Account:
#SBATCH --account=fc_contact
#
# Partition:
#SBATCH --partition=savio3
#
# Wall clock limit:
#SBATCH --time=00:00:30

#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (usually 1 for single-node)
#SBATCH --time=0:02:00             # Maximum runtime in HH:MM:SS


# Load the Conda module 
module load Anaconda3/23.5.2

# Activate your Conda environment
conda activate ModifiedCALenv

python /global/home/users/ogata/deepCAL/MakeData_thinker_savio.py

# Deactivate the Conda environment (optional)
conda deactivate
