cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
RESULTS_FOLDER=temporal_awareness/Cognition/results/temporal_interval

MODEL_NAME=gpt-3.5-turbo-0125
device=3

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "CUDA_ID: $device"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Cognition.gen_ans_closesource \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Cognition.get_outdated_qa_cognition \
    ${RESULTS_FOLDER}/${MODEL_NAME}