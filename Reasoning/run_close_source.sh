cd /path/project/models_output
MODEL_NAME=gpt-4o
device=0
TASK_TYPE=accumulate_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}_20250210


echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "CUDA_ID: $device"
echo "TASK_TYPE: $TASK_TYPE"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_closesource  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    --task_type ${TASK_TYPE}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
    ${RESULTS_FOLDER}/${MODEL_NAME}