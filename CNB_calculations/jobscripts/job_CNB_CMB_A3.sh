#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=fat_genoa
#SBATCH --time=80:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
echo "========= Job started at `date` =========="

module load 2024
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load FlexiBLAS/3.4.4-GCC-13.3.0
export OMPI_MCA_pml=ucx

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

cd /gpfs/home5/abellan/class_repositories/CLASSpp_nuDecay/CNB_calculations

python3.9 -u compute_CNB_CMB_A3.py

echo "========= Job finished at `date` =========="
