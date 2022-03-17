#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -j y

#$ -m beas
#$ -l gpus=1
#$ -pe omp 16

echo "Option: $1"

echo "Start!"

cd /project/privknn/embedding-tests/
module load cuda/11.2

taskset --cpu-list 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 python3 ./inversion_bert.py \
	--do_lower_case True \
	--learning True \
	--read_files /projectnb/privknn/attack-data-trec \
	--encrypted_training True \
	--encrypted_tag $1

# time taskset --cpu-list 1,2,3,4,5,6,7,8 python3 ./inversion_bert.py --do_lower_case True --learning True --read_files /projectnb/privknn/attack-data 2>&1 | tee log-4.txt

echo "Done!"
