#!/bin/bash
# SBATCH --account=rrg-eugenium           # Prof Eugene
# SBATCH --cpus-per-task=16                # Ask for 16 CPUs
# SBATCH --gres=gpu:1                     # Ask for 1 GPU
# SBATCH --mem=32G                        # Ask for 32 GB of RAM
# SBATCH --time=12:00:00                  # The job will run for 12 hours
# SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH

cd $SLURM_TMPDIR
cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/get_activations/repaired.result.pth.tar ./
cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/get_activations/get_png.py ./
cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/get_activations/utils.py ./
ls

cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/bash/imagenet.sh ./
bash imagenet.sh
cd $SLURM_TMPDIR

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/myvirenv
source $SLURM_TMPDIR/myvirenv/bin/activate

pip install --no-index torch torchvision numpy scipy tqdm matplotlib

python get_png.py