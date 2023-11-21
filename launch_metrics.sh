#!/bin/bash
#!/usr/bin/bash

#SBATCH --job-name='metrics'
#SBATCH --partition=normal256
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

set=$1
step_min=$2
step_max=$3
step_indent=$4
n_samples=$5
lr=$6
bs=$7

python3 -u metric_tests_exec.py --expe_set=$set --list_min=$step_min --list_max=$step_max --list_step=$step_indent --n_samples=$n_samples --lr0=$lr --batch_sizes=$bs --variables=['rr','u','v','t2m']
