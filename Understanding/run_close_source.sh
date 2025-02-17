cd /path/project/models_output
MODEL_NAME=gpt-3.5-turbo-0125
device=2
TASK_TYPE=explict
RESULTS_FOLDER=temporal_awareness/Understanding/${TASK_TYPE}

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "CUDA_ID: $device"
echo "TASK_TYPE: $TASK_TYPE"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Understanding.gen_ans_closesource  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    --task_type ${TASK_TYPE}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Understanding.get_outdated_qa_Understanding \
    ${RESULTS_FOLDER}/${MODEL_NAME}