cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
RESULTS_FOLDER=temporal_awareness/Cognition/results/temporal_interval

LLM=Phi-4
MODEL_NAME=${LLM}-Instruct
device=3
# LLM=Llama-2-7b, Llama-2-70b, Llama-3.1-8B, Llama-3.1-70B, Qwen2-7B, Qwen2-72B

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "CUDA_ID: $device"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Cognition.gen_ans_opensource \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Cognition.get_outdated_qa_cognition \
    ${RESULTS_FOLDER}/${MODEL_NAME}