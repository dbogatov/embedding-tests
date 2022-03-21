#!/bin/bash -l

#$ -l h_rt=48:00:00
#$ -j y

#$ -m beas
#$ -l gpus=1
#$ -pe omp 8

echo "Option: $1"

echo "Start!"

cd /project/privknn/embedding-tests/
module load cuda/11.2

while getopts "e:m:t:c:n:" o
do
	case "${o}" in
		e)
			EPOCHS="--epochs ${OPTARG}"
			;;
		m)
			READ_MODEL="--read_model_epoch ${OPTARG}"
			;;
		t)
			TREC="${OPTARG}"
			;;
		c)
			ENCn_TAG="--encrypted_tag ${OPTARG}"
			;;
		n)
			ENC_TRAINING="--encrypted_training ${OPTARG}"
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
	$EPOCHS \
	$READ_MODEL \
	$ENC_TAG \
	$ENC_TRAINING

# command 2>&1 | tee log-$(date +%m-%d-%Y-%H-%M-%S).txt

echo "Done!"