cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
RESULTS_FOLDER=temporal_awareness/Trustworthiness/past_unanswerable_date_copy
MODEL_NAME=gpt-3.5-turbo-0125
device=3

USE_TIMESTAMP_FLAG=""
USE_PAST_FLAG=""

if [[ "$RESULTS_FOLDER" == *"date"* ]]; then
    USE_TIMESTAMP_FLAG="--use_timestamp"
fi

if [[ "$RESULTS_FOLDER" == *"past"* ]]; then
    USE_PAST_FLAG="--use_past"
fi

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "USE_TIMESTAMP_FLAG: $USE_TIMESTAMP_FLAG"
echo "USE_PAST_FLAG: $USE_PAST_FLAG"
echo "CUDA_ID: $device"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Trustworthiness.gen_ans_closesource  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    ${USE_TIMESTAMP_FLAG} \
    ${USE_PAST_FLAG}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Trustworthiness.get_outdated_qa_trust \
    ${RESULTS_FOLDER}/${MODEL_NAME}