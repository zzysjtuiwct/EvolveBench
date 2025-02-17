cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context
#timetravel_outdate_date_w_up2date_context
LLM=Phi-4
MODEL_NAME=${LLM}-Instruct
device=2
# LLM=Llama-2-7b, Llama-2-70b, Llama-3.1-8B, Llama-3.1-70B, Qwen2-7B, Qwen2-72B
USE_RAG_FLAG=""
USE_TIMESTAMP_FLAG=""
USE_TIMETRAVEL_FLAG=""
if [[ "$RESULTS_FOLDER" == *"context"* ]]; then
    USE_RAG_FLAG="--use_rag"
fi

if [[ "$RESULTS_FOLDER" == *"date"* ]]; then
    USE_TIMESTAMP_FLAG="--use_timestamp"
fi

if [[ "$RESULTS_FOLDER" == *"timetravel"* ]]; then
    USE_TIMETRAVEL_FLAG="--if_timetravel"
fi

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "USE_RAG_FLAG: $USE_RAG_FLAG"
echo "USE_TIMESTAMP_FLAG: $USE_TIMESTAMP_FLAG"
echo "USE_TIMETRAVEL_FLAG: $USE_TIMETRAVEL_FLAG"
echo "CUDA_ID: $device"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_open_source  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    ${USE_RAG_FLAG} \
    ${USE_TIMESTAMP_FLAG} \
    ${USE_TIMETRAVEL_FLAG}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
    ${RESULTS_FOLDER}/${MODEL_NAME}

cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context_rag
LLMS=("Llama-3.1-70B" "Qwen2.5-72B")
device=0,1
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
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
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_open_source  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        ${USE_RAG_FLAG} \
        ${USE_TIMESTAMP_FLAG} \
        ${USE_TIMETRAVEL_FLAG} \
        ${USE_REAL_RAG_CONTEXT}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done


cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context_rag
LLMS=("Llama-3-70B" "Llama-3.3-70B")
device=2,3
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
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
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_open_source  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        ${USE_RAG_FLAG} \
        ${USE_TIMESTAMP_FLAG} \
        ${USE_TIMETRAVEL_FLAG} \
        ${USE_REAL_RAG_CONTEXT}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done

cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context_rag
LLMS=("Qwen2-72B" "Llama-2-70b")
device=2,3
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
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
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_open_source  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        ${USE_RAG_FLAG} \
        ${USE_TIMESTAMP_FLAG} \
        ${USE_TIMETRAVEL_FLAG} \
        ${USE_REAL_RAG_CONTEXT}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done

cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context_rag
LLMS=("Llama-2-7b" "Llama-2-13b" "Llama-3-8B")
device=1
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
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
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_open_source  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        ${USE_RAG_FLAG} \
        ${USE_TIMESTAMP_FLAG} \
        ${USE_TIMETRAVEL_FLAG} \
        ${USE_REAL_RAG_CONTEXT}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done


cd /path/project/models_output
RESULTS_FOLDER=temporal_awareness/Awareness/timetravel_outdate_date_w_up2date_context_rag
LLMS=("Llama-3.1-8B" "Phi-4" "Qwen2-7B" "Qwen2.5-7B")
device=0
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
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
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.generate_answers_open_source  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        ${USE_RAG_FLAG} \
        ${USE_TIMESTAMP_FLAG} \
        ${USE_TIMETRAVEL_FLAG} \
        ${USE_REAL_RAG_CONTEXT}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Awareness.get_outdated_questions \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done