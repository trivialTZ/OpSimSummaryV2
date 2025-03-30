#!/bin/bash
#SBATCH -J LSSTSIMLIB_wHost
#SBATCH -A m1727
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --array=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70GB
#SBATCH -e ./logs/LSSTSIMLIB_%a.err
#SBATCH -o ./logs/LSSTSIMLIB_%a.out

# Initialize conda properly
eval "$(conda shell.bash hook)"
conda activate opsim

HOST_FILE=/Users/tz/PycharmProjects/OpSimSummaryV2/scripts/gal_par/UchuuDR2_UM_z0p00_zmax0p1739_mock_00_SNANA.parquet

python make_simlib.py \
    'debass_fake_new.db' \
    --host_file $HOST_FILE \
    --output_dir 'simlibs\ceshi_' \
    --author 'tz' \
    --n_cpu 1
