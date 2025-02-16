import argparse
import json, ipdb
import os
import re
from argparse import Namespace
from typing import Dict, List, Optional, Set
from datetime import datetime

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import spacy
from spacy.tokenizer import Tokenizer
from spacy.tokens.doc import Doc
from spacy.tokens import Span
from spacy.language import Language
from tqdm import tqdm

from utils import write_roman, EXCEPTIONS, ADDITIONAL_BITS, load_json, dump_json


MONARCH_NUMS = {write_roman(i) for i in range(1, 100, 1)}


def extract_category(file_to_analyze: str):
    return "_".join(file_to_analyze.split("_")[:-1])

def parse_date(date_str):
    return datetime.strptime(date_str, "+%Y-%m-%dT%H:%M:%SZ")

def is_exception(
    answer_name: str,
    category: str,
    element: str,
    attribute: Optional[str],
    exceptions: dict,
) -> bool:
    if category in exceptions:
        if element in exceptions[category]:
            if attribute is None:
                return answer_name in exceptions[category][element]
            else:
                if attribute in exceptions[category][element]:
                    return answer_name in exceptions[category][element][attribute]

    return False

# 构造的数据集应该要保证每一个item都有start时间否则排序无法保证先后顺序
def extract_answer(
    answers: List[str],
    exceptions: dict,
    category: str,
    element: str,
    attribute: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    to_assign = {}

    no_end_entries = []
    answers = list(dict.fromkeys(answers)) # eliminate duplication
    # Sort the list so that we consider only the latest entry for a given candidate
    for answer in sorted(answers):#这里似乎可以做到同一姓名的可以以开始时间从小到大排序
        # Skip the answer if it is an exception

        split_answer = answer.split("|")
        name, span = split_answer[0], split_answer[1:]
        name = name.strip()  # remove first and last spaces

        if is_exception(name, category, element, attribute, exceptions):
            continue

        assert len(span) <= 2, "Additional elements in the answer span"

        if len(span) == 2:
            start, end = span
            start, end = re.sub("-00", "-01", start[3:].strip()), re.sub( #似乎很多数据的时间都是精确到年份，在原始数据中，月份和日期都是00，这里是把00换成了01
                "-00", "-01", end[3:].strip()
            )
        else:
            single_span = span.pop()
            if single_span.startswith("S:"):
                start, end = re.sub("-00", "-01", single_span[3:].strip()), None
                no_end_entries.append(answer)
            elif single_span.startswith("E:"):
                start, end = None, re.sub("-00", "-01", single_span[3:].strip())

        if name in to_assign:
            start_date = parse_date(start)
            prev_start = parse_date(to_assign[name]["start"])
            prev_end = parse_date(to_assign[name]["end"])
            if end is not None:
                end_date = parse_date(end)
                # 如果保证每个item都有start date，那么排序后，前后两个同name的时间区间仅有两种可能情况需要考虑。
                if end_date <= prev_end and start_date >= prev_start: #确实存在一些case，在排序后会出现当前end_date早于先前记录的情况，不知道作者是怎么发现的
                    start, end = to_assign[name]["start"], to_assign[name]["end"]
                elif prev_start < start_date and start_date <= prev_end and prev_end <= end_date:
                    start = to_assign[name]["start"]
            else:
                # 如果没有 `end` 且区间部分重叠
                if prev_start < start_date and start_date <= prev_end:
                    start = to_assign[name]["start"]
        
        to_assign[name] = {"start": start, "end": end} #字典，使得最终只保留最新的，即所有开始时间的最大值和所有结束时间的最大值

    assert (
        len(no_end_entries) == 1
    ), f"There are {len(no_end_entries)} entries with no end for {category} {element} {attribute if attribute else ''}: {no_end_entries}"

    return to_assign


def prepare_answers(category: str, original: dict, exceptions: dict) -> dict:
    answers = {}

    if category in ["countries_byGDP", "organizations"]:
        for element, attributes in original[category].items():
            if element not in answers: #似乎这个判断有点无用
                answers[element] = {}
            for attribute, grc_elem in attributes.items():
                to_assign = extract_answer(
                    grc_elem["answers"], exceptions, category, element, attribute
                )
                answers[element][attribute] = to_assign
    else:
        for element, grc_elem in original[category].items():
            to_assign = extract_answer(
                grc_elem["answers"], exceptions, category, element
            )
            answers[element] = to_assign

    return answers


def prepare_predictions(generated: dict, category: str, use_rfind: bool) -> dict:
    predictions = {}

    if category in ["countries_byGDP", "organizations"]:
        for element, attributes in generated.items():
            if element not in predictions:
                predictions[element] = {}

            for attribute, gen_elem in attributes.items():
                for qt, answer in gen_elem["answers"].items():
                    if attribute not in predictions[element]:
                        predictions[element][attribute] = {}
                    predictions[element][attribute][qt] = answer
    else:
        for element, gen_elem in generated.items():
            if element not in predictions:
                predictions[element] = {}
            for qt, answer in gen_elem["answers"].items():
                predictions[element][qt] = answer

    return predictions


def find_main_chunk(doc: Doc):
    ancestor = None
    for chunk in doc.noun_chunks: # doc.noun_chunks：spaCy 提供的一个生成器，用于提取文档中的名词短语。
        if ancestor is None:
            ancestor = chunk
        elif chunk.root.is_ancestor(ancestor.root):
            ancestor = chunk.root
    return ancestor


def is_monarch(span: Span, monarch_nums: Set[str]):
    for name_chunk in span.text.split():
        if name_chunk in monarch_nums:
            return True
    return False


def remove_additional_bits(string: str, additional_bits: List[str]):
    for bit in additional_bits:
        string = re.sub(bit, "", string)
    return " ".join(string.split())  # remove additional whitespaces


def assign_question_to_group_based_on_answer(
    stats: dict,
    question_type: str,
    pred: str,
    ans: Dict[str, Dict[str, str]],
    nlp: Language,
    monarch_nums: Set[str],
    additional_bits: List[str],
    element: str,
    attribute: Optional[str] = None,
    ground_truth: str = None,
):
    if question_type not in stats:
        stats[question_type] = {
            "correct": [],
            "irrelevant": [],
            "other": [],
        }

    to_append = {
        "prediction": pred,
        "matched_answers": [],
        "match_type": [],
        "correct_answer": ground_truth,
        "element": element,
        "attribute": attribute,
    }

    def check_match_position( #没太理解为什么下面的if判断要这么做，是为了同一个模型回复中生成了两个答案吗？
        match_span_idx: int,
        start_idx: int,
        to_append: dict,
        match_type: str,
        matches: int,
        is_correct: bool,
        answer: str,
    ):
        # If the current match occurs earlier than the other matches, discard
        # the previous matches (i.e. reset everything)
        # This means if a answer match starts at a earlier point in the prediction than previous answers, than it means a 
        # more complete match between the pred and the answer (The assumption is valid when the model follow the instruction 
        # that only output the name without any other sentence). Than take the better and discard the previous.
        if start_idx < match_span_idx:
            match_span_idx = start_idx
            to_append.update(
                {
                    "matched_answers": [],
                    "match_type": [],
                }
            )
            matches = 0
            is_correct = False

        match = False
        matched_ans = None
        # If the current match occurs at the same position of the other macthes,
        # add it to the pool of correct matches
        if start_idx == match_span_idx:
            match = True
            matched_ans = answer
            to_append["match_type"].append(match_type)

        return match, matches, match_span_idx, is_correct, matched_ans

    matches = 0
    is_correct = False
    match_span_idx = float("Inf")

    for answer, ans_prop in ans.items():
        match = False

        res = re.search(
            f"(^|[^\w]{{1}}){answer}($|[^\w]{{1}})", pred, flags=re.IGNORECASE
        )
        if bool(res):
            start_idx = res.span()[0]
            match, matches, match_span_idx, is_correct, matched_ans = (
                check_match_position(
                    match_span_idx,
                    start_idx,
                    to_append,
                    "em",
                    matches,
                    is_correct,
                    answer,
                )
            )
        else:
            answer = remove_additional_bits(answer, additional_bits)

            # try to see if you can match the simplified version
            res = re.search(
                f"(^|[^\w]{{1}}){answer}($|[^\w]{{1}})", pred, flags=re.IGNORECASE
            )
            if bool(res):
                start_idx = res.span()[0]
                match, matches, match_span_idx, is_correct, matched_ans = (
                    check_match_position(
                        match_span_idx,
                        start_idx,
                        to_append,
                        "simplified",
                        matches,
                        is_correct,
                        answer,
                    )
                )

            # check if we are considering a single token
            elif len(answer.split()) > 1:
                doc = nlp(answer)
                main_chunk = find_main_chunk(doc)

                if main_chunk is not None:
                    if is_monarch(main_chunk, monarch_nums):
                        head_chunk = main_chunk.text # e.g. Charles III
                    else:
                        head_chunk = main_chunk.root.text # e.g. III
                else:
                    head_chunk = answer

                res = re.search(
                    f"(^|[^\w]{{1}}){head_chunk}($|[^\w]{{1}})",
                    pred,
                    flags=re.IGNORECASE,
                )
                if bool(res):
                    start_idx = res.span()[0]
                    match, matches, match_span_idx, is_correct, matched_ans = (
                        check_match_position(
                            match_span_idx,
                            start_idx,
                            to_append,
                            "head",
                            matches,
                            is_correct,
                            head_chunk,
                        )
                    )

        if match:
            matches += 1
            to_append["matched_answers"].append((matched_ans, ans_prop["start"], ans_prop["end"]))
            if answer == to_append["correct_answer"]:
                is_correct = True
    
    if matches == 1 and is_correct:
        stats[question_type]["correct"].append(to_append)
    elif matches == 1 and not is_correct:
        stats[question_type]["other"].append(to_append)
    elif matches == 0:
        stats[question_type]["irrelevant"].append(to_append)
    elif matches > 1: # if matched more than one, then we chose the answer with longest length.
        longest_answer = max(to_append["matched_answers"], key=lambda x: len(x[0]))
        to_append["matched_answers"] = [x for x in to_append["matched_answers"] if x == longest_answer]
        if longest_answer[0] == to_append["correct_answer"]:
            stats[question_type]["correct"].append(to_append)
        else:
            stats[question_type]["other"].append(to_append)
    else:
        raise "I forgot to consider some cases"


def compute_stats_for_qa(
    predictions: dict,
    answers: dict,
    category: str,
    nlp: Language,
    monarch_nums: Set[str],
    additional_bits: Dict[str, List[str]],
    original_qa: dict
) -> Dict[str, Dict[str, dict]]:
    stats = {}

    # pass empty list if we are not considering athletes to avoid unwanted substitutions
    if category in ["countries_byGDP", "organizations"]:
        for (p_element, p_attributes), (a_attributes) in tqdm(
            list(zip(predictions.items(), answers.values())), desc=category
        ):
            for (p_attribute, questions), (ans) in zip(
                p_attributes.items(), a_attributes.values()
            ):
                for (
                    question_type,
                    pred,
                ) in questions.items():
                    assign_question_to_group_based_on_answer(
                        stats,
                        question_type,
                        pred,
                        ans,
                        nlp,
                        monarch_nums,
                        additional_bits.get(category, []),
                        p_element,
                        p_attribute,
                        original_qa[category][p_element][p_attribute]['ground_truth']
                    )
    else:
        for (p_element, questions), (ans) in tqdm(
            list(zip(predictions.items(), answers.values())), desc=category
        ):
            for question_type, pred in questions.items():
                assign_question_to_group_based_on_answer(
                    stats,
                    question_type,
                    pred,
                    ans,
                    nlp,
                    monarch_nums,
                    additional_bits.get(category, []),
                    p_element,
                    None,
                    original_qa[category][p_element]['ground_truth']
                )
    return stats


def save_stats(stats: dict, category: str, results_folder: str, indent: int = 4):
    path = os.path.join(results_folder, f"{category}_analysis.json")
    dump_json(path, stats, indent)


def analyze_model_replies(
    results_folder: str,
    questions_path: str,
    nlp: Language,
    monarch_nums: Set[str],
    additional_bits: Dict[str, List[str]],
    exceptions: dict,
    use_rfind: bool,
):
    total_stats = {}
    for domain_dir, dirs, files in os.walk(results_folder):
        for file_to_analyze in files:
            if file_to_analyze.endswith("_answers.json"):
                category = extract_category(file_to_analyze)
                stats_path = os.path.join(
                    domain_dir,
                    "".join(
                        file_to_analyze.split("_answers.json")[:-1] + ["_analysis.json"]
                    ),
                )
                if not os.path.isfile(stats_path):
                    generated = load_json(os.path.join(domain_dir, file_to_analyze))
                    original = load_json(questions_path)

                    answers = prepare_answers(category, original, exceptions)
                    predictions = prepare_predictions(generated, category, use_rfind)
                    
                    stats = compute_stats_for_qa(
                        predictions, answers, category, nlp, monarch_nums, additional_bits, original
                    )
                    save_stats(stats, category, domain_dir)
                    total_stats[category] = stats
                else:
                    total_stats[category] = load_json(os.path.join(domain_dir, f"{category}_analysis.json"))
                    print(f"File {stats_path} already exists: SKIPPING")
    
    dump_json(os.path.join(results_folder, "analysis.json"), total_stats, indent=4)

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m analyze_replies",
        description="Analyze the generated answers of a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "results_dir",
        metavar="DIR_NAME",
        type=str,
        help="Folder containing the generated answers from a model.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def analyze_replies(results_folder: str, questions_path: str):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = Tokenizer(nlp.vocab)  # Whitespace tokenization

    analyze_model_replies(
        results_folder,
        questions_path,
        nlp,
        MONARCH_NUMS,
        ADDITIONAL_BITS,
        EXCEPTIONS,
        use_rfind=True
    )


if __name__ == "__main__":
    args = get_args()
    print("Analyzing the replies of the model...")
    analyze_replies(args.results_dir, args.question_path)
