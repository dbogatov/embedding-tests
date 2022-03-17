#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -j y

#$ -m beas
#$ -l gpus=1
#$ -pe omp 16

echo "Start!"

cd /project/privknn/embedding-tests/
module load cuda/11.2

for beta in {0..50}
do
	time taskset --cpu-list 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 python3 ./inversion_bert.py \
		--do_lower_case True \
		--learning True \
		--read_files /projectnb/privknn/attack-data-trec \
		--epochs 5 \
		--read_model_epoch 4 \
		--encrypted_tag encrypted-${beta}.0-
done

echo "Done!"
