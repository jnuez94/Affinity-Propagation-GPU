#!/bin/sh
var=$(pwd)
echo "The Current Working Directory : $var"
find . -name "slurm-*.out" -exec rm {} \;
sbatch --gres=gpu:1 --wrap="/opt/PYTHON/bin/python $var/$1" --time 1;
while  [ true ]
  do
        if ls slurm-* > /dev/null 2>&1
        then
                OldTimestamp=$(date -r slurm-*);
                NewTimestamp=$OldTimestamp;
                while [  "$NewTimestamp" = "$OldTimestamp" ];
                do
                        OldTimestamp=$(date -r slurm-*);
                done;
                break;
        fi
  done
echo "";
cat slurm-*.out;
echo "";
