#!/bin/bash
# SBATCH --account=rrg-eugenium           # Prof Eugene
# SBATCH --cpus-per-task=16                # Ask for 16 CPUs
# SBATCH --gres=gpu:1                     # Ask for 1 GPU
# SBATCH --mem=32G                        # Ask for 32 GB of RAM
# SBATCH --time=2:00:00                  # The job will run for 12 hours
# SBATCH -o /scratch/vs2410/slurm-%j.out  # Write the log in $SCRATCH


cd $SLURM_TMPDIR
# extract imagenet images
cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/bash/imagenet.sh ./
bash imagenet.sh

# copy models
echo "copy trained model parameters"
cp ~/projects/rrg-eugenium/cszou/experiment_results/matching/m* ./

echo "copy validation codes"
mkdir $SLURM_TMPDIR/weight_matching
cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/weight_matching/* ./weight_matching/

echo "copy REPAIR codes"
cp ~/projects/rrg-eugenium/cszou/matching_and_interpolation/REPAIR/* ./

# run interpolation code
echo "repair model"
python repair.py
cp ./wrap_a ~/projects/rrg-eugenium/cszou/matching_and_interpolation/REPAIR/
cp ./model_b ~/projects/rrg-eugenium/cszou/matching_and_interpolation/REPAIR/
cp ./modelMatched ~/projects/rrg-eugenium/cszou/matching_and_interpolation/REPAIR/
