#!/bin/bash


echo 'start'

PBS_TMPDIR=$TMPDIR
unset TMPDIR



if [ -z $config_file ]; then
  echo "configuration file mising"
  exit 1
fi
echo $config



module unload conda
module load conda
conda activate /softs/rh8/conda-envs/pangeo_stable

echo 'start get_swh_saturations.py'
python $PWD/CODE/app/get_swh_gkde.py -c $config_file 
                                             




