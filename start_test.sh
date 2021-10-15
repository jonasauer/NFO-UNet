#!/bin/bash
#SBATCH -J test
#SBATCH -N 1
#SBATCH --partition gpu_gtx1080single
#SBATCH --qos gpu_gtx1080single
#SBATCH --gres gpu:1
#SBATCH --mail-type=ALL    # first have to state the type of event to occur
#SBATCH --mail-user=e1426685@student.tuwien.ac.at

module purge
module load gcc/5.3 python/3.6 cuda/10.0.130
nvidia-smi

source ~/project/code/venv/bin/activate
cd ~/project/code
python3 -m test_main --config $1 --load-dir $2;


