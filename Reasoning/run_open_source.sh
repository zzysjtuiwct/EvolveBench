cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
LLM=Phi-4
MODEL_NAME=${LLM}-Instruct
device=5
TASK_TYPE=ranking_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}
# LLM=Llama-2-7b, Llama-2-70b, Llama-3.1-8B, Llama-3.1-70B, Qwen2-7B, Qwen2-72B

echo "MODEL_NAME: $MODEL_NAME"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "CUDA_ID: $device"
echo "TASK_TYPE: $TASK_TYPE"

# Open Book
CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_opensource  \
    ${MODEL_NAME} \
    --out-dir ${RESULTS_FOLDER} \
    --task_type ${TASK_TYPE}

CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
    ${RESULTS_FOLDER}/${MODEL_NAME}

cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
device=0,1
TASK_TYPE=accumulate_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}_20250210
LLMS=("Llama-3.1-70B" "Qwen2.5-72B")
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
    echo "MODEL_NAME: $MODEL_NAME"
    echo "RESULTS_FOLDER: $RESULTS_FOLDER"
    echo "CUDA_ID: $device"
    echo "TASK_TYPE: $TASK_TYPE"

    # Open Book
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_opensource  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        --task_type ${TASK_TYPE}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done

cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
device=2,3
TASK_TYPE=accumulate_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}_20250210
LLMS=("Llama-3-70B" "Llama-3.3-70B")
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
    echo "MODEL_NAME: $MODEL_NAME"
    echo "RESULTS_FOLDER: $RESULTS_FOLDER"
    echo "CUDA_ID: $device"
    echo "TASK_TYPE: $TASK_TYPE"

    # Open Book
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_opensource  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        --task_type ${TASK_TYPE}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done

cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
device=4,5
TASK_TYPE=accumulate_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}_20250210
LLMS=("Qwen2-72B" "Llama-2-70b")
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
    echo "MODEL_NAME: $MODEL_NAME"
    echo "RESULTS_FOLDER: $RESULTS_FOLDER"
    echo "CUDA_ID: $device"
    echo "TASK_TYPE: $TASK_TYPE"

    # Open Book
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_opensource  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        --task_type ${TASK_TYPE}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done

cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
device=6
TASK_TYPE=accumulate_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}_20250210
LLMS=("Llama-2-7b" "Llama-2-13b" "Llama-3-8B")
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
    echo "MODEL_NAME: $MODEL_NAME"
    echo "RESULTS_FOLDER: $RESULTS_FOLDER"
    echo "CUDA_ID: $device"
    echo "TASK_TYPE: $TASK_TYPE"

    # Open Book
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_opensource  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        --task_type ${TASK_TYPE}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done

cd /remote-home/zhiyuanzhu/project/DyKnow/models_output
device=7
TASK_TYPE=accumulate_qa
RESULTS_FOLDER=temporal_awareness/Reasoning/results/${TASK_TYPE}_20250210
LLMS=("Llama-3.1-8B" "Phi-4" "Qwen2-7B" "Qwen2.5-7B")
for LLM in "${LLMS[@]}"; do
    MODEL_NAME="${LLM}-Instruct"
    echo "Running for ${MODEL_NAME} on CUDA device ${DEVICE}..."
    echo "MODEL_NAME: $MODEL_NAME"
    echo "RESULTS_FOLDER: $RESULTS_FOLDER"
    echo "CUDA_ID: $device"
    echo "TASK_TYPE: $TASK_TYPE"

    # Open Book
    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.gen_ans_opensource  \
        ${MODEL_NAME} \
        --out-dir ${RESULTS_FOLDER} \
        --task_type ${TASK_TYPE}

    CUDA_VISIBLE_DEVICES=${device} python -m temporal_awareness.Reasoning.get_outdated_qa_reasoning \
        ${RESULTS_FOLDER}/${MODEL_NAME}

    echo "Completed processing for ${MODEL_NAME}."
done