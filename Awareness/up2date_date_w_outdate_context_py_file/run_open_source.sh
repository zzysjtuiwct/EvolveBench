cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/up2date_date_20250101_w_outdate_context
# up2date_date_20250101_w_outdate_context
LLM=Llama-3-70B
MODEL_NAME=${LLM}-Instruct
device=2,3
# LLM=Llama-2-7b, Llama-2-70b, Llama-3.1-8B, Llama-3.1-70B, Qwen2-7B, Qwen2-72B

USE_RAG_FLAG=""
USE_TIMESTAMP_FLAG=""
if [[ "$RESULTS_FOLDER" == *"context"* ]]; then
    USE_RAG_FLAG="--use_rag"
fi

if [[ "$RESULTS_FOLDER" == *"date"* ]]; then
    USE_TIMESTAMP_FLAG="--use_timestamp"
fi

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "USE_RAG_FLAG: $USE_RAG_FLAG"
echo "USE_TIMESTAMP_FLAG: $USE_TIMESTAMP_FLAG"
echo "CUDA_ID: $device"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m generate_rag_answers  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    ${USE_RAG_FLAG} \
    ${USE_TIMESTAMP_FLAG}

CUDA_VISIBLE_DEVICES=${device} python -m get_outdated_questions \
    ${RESULTS_FOLDER}/${MODEL_NAME}