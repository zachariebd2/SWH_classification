#!/bin/bash


echo 'start'


PBS_TMPDIR=$TMPDIR

unset TMPDIR







module unload conda

module load conda

conda activate /softs/rh8/conda-envs/s3-env

echo 'start get_swh_saturations.py'

echo $PWD/CODE/app/get_swh_saturations.py
echo -c $config_file
echo -t $PBS_TMPDIR 

python $PWD/CODE/app/get_swh_saturations.py -c $config_file \
                                          -t $PBS_TMPDIR 




