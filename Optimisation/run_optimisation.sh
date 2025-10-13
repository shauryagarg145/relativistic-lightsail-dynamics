#!/bin/bash

#PBS -P dg76
#PBS -q normalsr
#PBS -l ncpus=200
#PBS -l mem=200GB
#PBS -l walltime=24:30:00

module purge
module load use.own
module load python3/3.12.1
module load numpy/2.2.5
module load scipy/1.13.1
module load adaptive/1.3.0
module load autograd/1.8.0
module load grcwa/0.1.2
module load nlopt/2.9.1
module load torch/2.7.0
module load torcwa/0.1.4.2
module load multiprocess/0.70.18
module load dill/0.4.0

cd /scratch/dg76/sg5635/relativistic-lightsail-dynamics/Optimisation

python3 /scratch/dg76/sg5635/relativistic-lightsail-dynamics/Optimisation/run_parallel.py
