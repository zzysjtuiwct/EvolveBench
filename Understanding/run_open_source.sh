cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
LLM=Phi-4
MODEL_NAME=${LLM}-Instruct
device=1
TASK_TYPE=explict
RESULTS_FOLDER=temporal_awareness/Understanding/${TASK_TYPE}
# LLM=Llama-2-7b, Llama-2-70b, Llama-3.1-8B, Llama-3.1-70B, Qwen2-7B, Qwen2-72B

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "CUDA_ID: $device"
echo "TASK_TYPE: $TASK_TYPE"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Understanding.gen_ans_opensource  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    --task_type ${TASK_TYPE}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Understanding.get_outdated_qa_Understanding \
    ${RESULTS_FOLDER}/${MODEL_NAME}