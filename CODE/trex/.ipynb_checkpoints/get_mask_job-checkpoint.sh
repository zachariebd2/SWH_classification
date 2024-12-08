#!/bin/bash


echo 'start'

PBS_TMPDIR=$TMPDIR
unset TMPDIR



if [ -z $csv_file ]; then
  echo "csv file mising"
  exit 1
fi
echo $config

module unload conda
module load conda
conda activate /softs/rh8/conda-envs/pangeo_stable

echo 'start get_mask.py'
python $PWD/CODE/app/get_mask.py -c $csv_file




