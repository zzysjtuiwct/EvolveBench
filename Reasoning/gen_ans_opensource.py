import argparse,re
import json
import os, ipdb
from argparse import Namespace
from typing import Dict, List, Optional
import random
import numpy as np
import torch
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m generate_rag_answers",
        description="Generate answers to reasoning questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        choices={"mistral", "Llama-2-7b-Instruct", "Llama-2-13b-Instruct", "Llama-2-70b-Instruct", "Llama-3-8B-Instruct", "Llama-3-70B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Llama-3.3-70B-Instruct","Qwen2-7B-Instruct", "Qwen2-72B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct", "Phi-4-Instruct"},
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "--qa_file",
        metavar="QA_FILE",
        type=str,
        default="temporal_awareness/Reasoning/reasoning_qa_20250210.json",
        help="Path to the QA file containing model-specific outdated questions.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--max-length",
        metavar="LENGTH",
        type=int,
        default=128,
        help="Max number of tokens that can be generated.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="results",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--num-gpus",
        metavar="NUM-GPUs",
        type=int,
        default=2,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--task_type",
        metavar="TASK TYPE",
        choices={"ranking_qa", "accumulate_qa"},
        type=str,
        default="ranking_qa",
        help="Reasoning task type.",
    )
    
    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args
    

def encode_inputs(
        q: str,
        tokenizer: AutoTokenizer,
        model_name: str,
    ) -> BatchEncoding:

    system_prompt = "You are a knowledgeable assistant who can answer factual questions."

    usr_prompt = f"You should answer the question using your knowledge and reasoning capacity. Remember, your answer must contain only the name, with no other words.\n\n"
    context_prompt = f"QUESTION: {q}\n\nYour answer:"
    usr_prompt = usr_prompt + context_prompt

    if "Llama-2" in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            text, return_tensors='pt', padding=True
        )
    elif "Llama-3" in model_name:
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{usr_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = tokenizer(
            text, return_tensors="pt", padding=True
        )
    elif "Qwen2" in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            text, return_tensors="pt", padding=True
        )
    elif "Phi-4" in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            text, return_tensors="pt", padding=True
        )
    else:
        assert False, "Model name is not in the expected range"
    
    return inputs,text


def generate_answers(
        questions: List[str],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        args: Namespace,
    ) -> Dict[str, Dict[str, str]]:

    res = {
        "questions": {},
        "answers": {}
    }
    for qt, q in questions.items():
        inputs, text = encode_inputs(q, tokenizer, args.model_name)
        res["questions"][qt] = text
        
        inputs = inputs.to(args.device)
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_length,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        res["answers"][qt] = tokenizer.batch_decode(
            generate_ids[:, inputs.input_ids.size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    return res

def set_seed(seed):
    random.seed(seed)  # Python的随机数生成器
    np.random.seed(seed)  # NumPy的随机数生成器
    torch.manual_seed(seed)  # PyTorch的随机数生成器
    torch.cuda.manual_seed(seed)  # CUDA设备的随机数生成器
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU，设置所有GPU的随机数生成器
    torch.backends.cudnn.deterministic = True  # 确保结果的确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动优化

model_configs = {
    "mistral": {"path": "mistralai/Mistral-7B-Instruct-v0.1", "precision": None},
    "Llama-2-7b-Instruct": {"path": "/path/download_model/meta-llama/llama2/Llama-2-7b-chat-hf", "precision": torch.float16},
    "Llama-2-13b-Instruct": {"path": "/path/download_model/meta-llama/llama2/Llama-2-13b-chat-hf", "precision": torch.float16},
    "Llama-2-70b-Instruct": {"path": "/path/download_model/meta-llama/llama2/Llama-2-70b-chat-hf", "precision": torch.float16},
    "Llama-3-8B-Instruct": {"path": "/path/download_model/meta-llama/llama3/Meta-Llama-3-8B-Instruct", "precision": torch.bfloat16},
    "Llama-3-70B-Instruct": {"path": "/path/download_model/meta-llama/llama3/Meta-Llama-3-70B-Instruct", "precision": torch.bfloat16},
    "Llama-3.1-8B-Instruct": {"path": "/path/download_model/meta-llama/llama3/Llama-3.1-8B-Instruct", "precision": torch.bfloat16},
    "Llama-3.1-70B-Instruct": {"path": "/path/download_model/meta-llama/llama3/Meta-Llama-3.1-70B-Instruct", "precision": torch.bfloat16},
    "Llama-3.3-70B-Instruct": {"path": "/path/download_model/meta-llama/llama3/Llama-3.3-70B-Instruct", "precision": torch.bfloat16},
    "Qwen2-7B-Instruct": {"path": "/path/download_model/Qwen2-7B-Instruct", "precision": torch.bfloat16},
    "Qwen2-72B-Instruct": {"path": "/path/download_model/Qwen2-72B-Instruct", "precision": torch.bfloat16},
    "Qwen2.5-7B-Instruct": {"path": "/path/download_model/Qwen2.5/Qwen2.5-7B-Instruct", "precision": torch.bfloat16},
    "Qwen2.5-72B-Instruct": {"path": "/path/download_model/Qwen2.5/Qwen2.5-72B-Instruct", "precision": torch.bfloat16},
    "Phi-4-Instruct": {"path": "/path/download_model/phi-4", "precision": torch.bfloat16}
}

def main():
    args = get_args()
    print(f"Generating answers of {args.model_name} model")
    set_seed(42)

    config = model_configs.get(args.model_name)
    if config is None:
        raise NotImplementedError

    model_path = config["path"]
    precision_type = config["precision"]

    out_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype = precision_type, 
        attn_implementation="flash_attention_2"
    )
    print("model name:", os.path.basename(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open(args.qa_file, "r") as f:
        original_questions = json.load(f)

    answers = {}
    for domain in original_questions:
        if domain not in answers:
            answers[domain] = {}
        for element in tqdm(original_questions[domain], desc=domain):
            if element not in answers[domain]:
                answers[domain][element] = {}
            if domain in ["countries_byGDP", "organizations"]:
                for attribute in original_questions[domain][element]:
                    if attribute not in answers[domain][element]:
                        answers[domain][element][attribute] = {}
                    questions = original_questions[domain][element][attribute][args.task_type]
                    answers[domain][element][attribute] = generate_answers(
                        questions, model, tokenizer, args
                    )
            else:
                questions = original_questions[domain][element][args.task_type]

                answers[domain][element] = generate_answers(
                    questions, model, tokenizer, args
                )

    for domain in answers:
        domain_dir = os.path.join(out_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        with open(os.path.join(domain_dir,  f"{domain}_answers.json"), "w") as f:
            json.dump(answers[domain], f, indent=4)

    with open(os.path.join(out_dir,  "answers.json"), "w") as f:
            json.dump(answers, f, indent=4)

if __name__ == "__main__":
    main()