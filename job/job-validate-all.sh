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

MAX_BETA=50
STEP=1.0

while getopts "s:b:t:" o
do
	case "${o}" in
		b)
			MAX_BETA="${OPTARG}"
			;;
		s)
			STEP="${OPTARG}"
			;;
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

for beta in $(seq -f "%.1f" 0.0 $STEP $MAX_BETA)
do
	time taskset --cpu-list 1,2,3,4,5,6,7,8 python3 ./inversion_bert.py \
		--do_lower_case True \
		--learning True \
		--read_files /projectnb/privknn/attack-data$TREC \
		--validation_only True \
		--encrypted_tag encrypted-${beta}-
done

echo "Done!"
