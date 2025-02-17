cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context_rag
# parameter_knowledge_w_prompt outdate_context_w_prompt
MODEL_NAME=gpt-3.5-turbo-0125
device=4

USE_RAG_FLAG=""
USE_TIMESTAMP_FLAG=""
USE_TIMETRAVEL_FLAG=""
USE_REAL_RAG_CONTEXT=""
if [[ "$RESULTS_FOLDER" == *"context"* ]]; then
    USE_RAG_FLAG="--use_rag"
fi

if [[ "$RESULTS_FOLDER" == *"date"* ]]; then
    USE_TIMESTAMP_FLAG="--use_timestamp"
fi

if [[ "$RESULTS_FOLDER" == *"timetravel"* ]]; then
    USE_TIMETRAVEL_FLAG="--if_timetravel"
fi

if [[ "$RESULTS_FOLDER" == *"rag"* ]]; then
    USE_REAL_RAG_CONTEXT="--if_context_from_rag"
fi
echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "USE_RAG_FLAG: $USE_RAG_FLAG"
echo "USE_TIMESTAMP_FLAG: $USE_TIMESTAMP_FLAG"
echo "USE_TIMETRAVEL_FLAG: $USE_TIMETRAVEL_FLAG"
echo "USE_REAL_RAG_CONTEXT: $USE_REAL_RAG_CONTEXT"
echo "CUDA_ID: $device"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_close_source  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    ${USE_RAG_FLAG} \
    ${USE_TIMESTAMP_FLAG} \
    ${USE_TIMETRAVEL_FLAG} \
    ${USE_REAL_RAG_CONTEXT}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
    ${RESULTS_FOLDER}/${MODEL_NAME}