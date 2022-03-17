#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -j y

#$ -m beas
#$ -l gpus=1
#$ -pe omp 8

echo "TREC or not: $1"

echo "Start!"

cd /project/privknn/embedding-tests/
module load cuda/11.2

time taskset --cpu-list 1,2,3,4,5,6,7,8 python3 ./inversion_bert.py \
	--do_lower_case True \
	--learning True \
	--epochs 10 \
	--read_model_epoch 4 \
	--read_files /projectnb/privknn/attack-data$1

# time taskset --cpu-list 1,2,3,4,5,6,7,8 python3 ./inversion_bert.py --do_lower_case True --learning True --read_files /projectnb/privknn/attack-data 2>&1 | tee log-$(date +%m-%d-%Y-%H-%M-%S).txt

echo "Done!"
