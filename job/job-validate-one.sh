#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -j y

#$ -m beas
#$ -l gpus=1
#$ -pe omp 8

echo $@

echo "Start!"

cd /project/privknn/embedding-tests/
module load cuda/11.2

while getopts "t:" o
do
	case "${o}" in
		t)
			TREC="${OPTARG}"
			;;
		*)
			echo "Problem with CLI"
			exit 1
			;;
	esac
done
shift $((OPTIND-1))

time taskset --cpu-list 1,2,3,4,5,6,7,8 python3 ./inversion_bert.py \
	--do_lower_case True \
	--learning True \
	--read_files /projectnb/privknn/attack-data$TREC \
	--validation_all True

echo "Done!"
