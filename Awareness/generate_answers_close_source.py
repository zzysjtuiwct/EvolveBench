import argparse
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
from openai import OpenAI

shared_GPT4_client = OpenAI(api_key="api-key")


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
        description="Generate answers to questions using RAG with a specific model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        choices={"gpt-4o","gpt-4o-mini", "gpt-3.5-turbo-0125"},
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "--qa_file",
        metavar="QA_FILE",
        type=str,
        default="some_path",
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
        "--passages-path",
        metavar="FILE_PATH",
        type=str,
        default="some_path",
        help="Path to the file containing the passages collected from Wikipedia.",
    )
    parser.add_argument(
        "--num-gpus",
        metavar="NUM-GPUs",
        type=int,
        default=2,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="weather to use RAG context",
    )
    parser.add_argument(
        "--use_timestamp",
        action="store_true",
        help="weather to use timestamp in prompt",
    )
    parser.add_argument(
        "--if_timetravel",
        action="store_true",
        help="weather to use timestamp in prompt",
    )
    parser.add_argument(
        "--if_context_from_rag",
        action="store_true",
        help="weather to use real rag results",
    )
    
    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args
    

def encode_inputs(
        date: str,
        q: str,
        context: str,
        model_name: str,
        if_rag: bool,
        if_timestamp: bool
    ) -> list:
    time_stamp_prompt = f"Today Date: {date}\n\n" if if_timestamp else ""

    system_prompt = "You are a knowledgeable assistant who can answer factual questions."

    if if_rag:
        usr_prompt = f"Given a question and its relevant context, you should answer it using your own knowledge or the knowledge provided by the context based on today's date ({date}). Remember, the provided context may not necessarily belong to the current era, and your answer must contain only the name, with no other words.\n\n"
        context_prompt = f"CONTEXT: {context}\n\nQUESTION: {q}?\n\nYour answer:"
        usr_prompt = time_stamp_prompt + usr_prompt + context_prompt
    else:
        usr_prompt = f"Given a question, you should answer it using your own knowledge based on today's date ({date}). Remember, your answer must contain only the name, with no other words.\n\n"
        context_prompt = f"QUESTION: {q}?\n\nYour answer:"
        usr_prompt = time_stamp_prompt + usr_prompt + context_prompt

    if "gpt" in model_name:
        messages=[
                {   
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": usr_prompt
                }
        ]
        # ipdb.set_trace()
    else:
        raise NotImplementedError
    
    return messages

def generate_answers(
        date: str,
        questions: List[str],
        client: OpenAI,
        args: Namespace,
        context: str,
        if_rag: bool,
        if_timestamp: bool
    ) -> Dict[str, Dict[str, str]]:

    res = {
        "questions": {},
        "answers": {}
    }
    for qt, q in questions.items():
        messages = encode_inputs(date, q, context, args.model_name, if_rag, if_timestamp)
        res["questions"][qt] = messages

        completion = None
        while not completion:
            try:
                completion = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages
                )
            except KeyboardInterrupt:
                print("Ctrl+C received! Exiting...")
                exit(0)
            except Exception as e:
                print(e)
                continue

        res["answers"][qt] = completion.choices[0].message.content.strip()

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
    "gpt-4o": shared_GPT4_client,
    "gpt-4o-mini": shared_GPT4_client,
    "gpt-3.5-turbo-0125": shared_GPT4_client
}

def main():
    args = get_args()
    print(f"Generating answers of {args.model_name} model")
    set_seed(42)

    client = model_configs.get(args.model_name)
    if client is None:
        raise NotImplementedError

    out_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    print("model name:", args.model_name)

    with open(args.qa_file, "r") as f:
        original_questions = json.load(f)

    with open(args.passages_path, "r") as f:
        passages = json.load(f)

    if args.if_timetravel:
        with open('some_path', "r") as f:
            time_tavel_info = json.load(f)

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
                    questions = original_questions[domain][element][attribute]["questions"]
                    question_passages = passages[domain][element][attribute]

                    outdated_context = question_passages["matches"]["outdate_knowledge"]["text"]
                    up2dated_context = question_passages["matches"]["up2date_knowledge"]["text"]

                    context = up2dated_context if args.if_timetravel else outdated_context
                    if args.if_context_from_rag:
                        context = question_passages["matches"]["rag_results"]["text"]
                    date = time_tavel_info[domain][element][attribute]['time_travel_date']
                    assert context == time_tavel_info[domain][element][attribute]['future_news']['text']

                    answers[domain][element][attribute] = generate_answers(
                        date, questions, client, args, context, if_rag=args.use_rag, if_timestamp=args.use_timestamp
                    )
            else:
                questions = original_questions[domain][element]["questions"]
                question_passages = passages[domain][element]

                outdated_context = question_passages["matches"]["outdate_knowledge"]["text"]
                up2dated_context = question_passages["matches"]["up2date_knowledge"]["text"]

                context = up2dated_context if args.if_timetravel else outdated_context
                if args.if_context_from_rag:
                    context = question_passages["matches"]["rag_results"]["text"]
                date = time_tavel_info[domain][element]['time_travel_date']
                assert context == time_tavel_info[domain][element]['future_news']['text']

                answers[domain][element] = generate_answers(
                    date, questions, client, args, context, if_rag=args.use_rag, if_timestamp=args.use_timestamp
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