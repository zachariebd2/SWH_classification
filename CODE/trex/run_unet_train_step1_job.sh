#!/bin/bash


echo 'start'




if [ -z $config_file ]; then
  echo "configuration file mising"
  exit 1
fi
echo $config

module unload conda
module load conda
conda activate /work/scratch/env/$USER/conda_env/pl_lightning

echo 'start unet_train_step1.py'
python $PWD/CODE/app/unet_train_step1.py -c $config_file \
                                               -tmp $TMPDIR




