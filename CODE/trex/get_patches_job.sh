#!/bin/bash


echo 'start'




if [ -z $config_file ]; then
  echo "configuration file mising"
  exit 1
fi
echo $config

module unload conda
module load conda
conda activate /softs/rh8/conda-envs/pangeo_stable

echo 'start get_patches.py'
python $PWD/CODE/app/get_patches.py -c $config_file \
                                               -tmp $TMPDIR




