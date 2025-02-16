import argparse, ipdb
import os
from argparse import Namespace

from utils import load_json, dump_json
from temporal_awareness.Cognition.analyze_replies_cognition import analyze_replies
from typing import Dict, List


def save_answer_sheet(results_folder: str, question_path: str):
    """
    Save the answer sheet based on the analysis of replies.

    Args:
        results_folder (str): The path to the folder containing the generated answers.
        question_path (str): The path to the question file.

    Returns:
        None
    """
    # make sure the _analysis.json files are there
    analyze_replies(results_folder, question_path)

    for domain_dir, dirs, files in os.walk(results_folder):
        for file_to_analyze in files:
            if file_to_analyze.endswith("_analysis.json"):
                analysis = load_json(os.path.join(domain_dir, file_to_analyze))

                answer_sheet = {}
                for question_type in analysis:
                    for answer_type, answers in analysis[question_type].items():
                        for ans in answers:
                            category = "_".join(file_to_analyze.split("_")[:-1])
                            element = ans["element"]
                            attribute = ans["attribute"]

                            if element not in answer_sheet:
                                answer_sheet[element] = {}

                            if attribute is None:
                                answer_sheet[element][question_type] = answer_type
                            else:
                                if attribute not in answer_sheet[element]:
                                    answer_sheet[element][attribute] = {}
                                answer_sheet[element][attribute][
                                    question_type
                                ] = answer_type

                dump_json(
                    os.path.join(domain_dir, f"{category}_answer_sheet.json"),
                    answer_sheet,
                )


def save_questions_to_update(results_folder: str, questions_path: str):
    """
    Save questions to update based on the answer sheets.

    Args:
        results_folder (str): The path to the folder containing the answer sheets.
        questions_path (str): The path to the questions file.

    Returns:
        None
    """
    save_answer_sheet(results_folder, questions_path)

    questions_non_correct = {}
    n_questions = 0
    for domain_dir, dirs, files in os.walk(results_folder):
        for answer_sheet_file in files:
            if answer_sheet_file.endswith("_answer_sheet.json"):
                questions = load_json(questions_path)
                answer_sheet = load_json(os.path.join(domain_dir, answer_sheet_file))
                category = "_".join(answer_sheet_file.split("_")[:-2])

                if category not in questions_non_correct:
                    questions_non_correct[category] = {}

                for element in answer_sheet:
                    if category not in ["countries_byGDP", "organizations"]:
                        answer_types = [
                            ans
                            for qt, ans in answer_sheet[element].items()
                            if qt not in ["generic"]
                        ]
                        assert len(answer_types) == 3, "I am not removing stuff"
                        if "match_correct_answer" not in answer_types:
                            if all([x == "irrelevant" for x in answer_types]):
                                continue
                            elif all(
                                [x == "match_other_answer" or x == "irrelevant" for x in answer_types]
                            ):
                                if element not in questions_non_correct[category]:
                                    questions_non_correct[category][element] = {}

                                questions_non_correct[category][element] = questions[
                                    category
                                ][element]
                                n_questions += 1
                                assert (
                                    "match_other_answer" in answer_types 
                                ), "No other answer matched in question types"
                            else:
                                raise AssertionError(
                                    f"Not covering case for {element}: '{answer_types}'"
                                )
                    else:
                        for attribute in answer_sheet[element]:
                            answer_types = [
                                ans
                                for qt, ans in answer_sheet[element][attribute].items()
                                if qt not in ["generic"]
                            ]
                            assert len(answer_types) == 3, "I am not removing stuff"
                            if "match_correct_answer" not in answer_types:
                                if all([x == "irrelevant" for x in answer_types]):
                                    continue
                                elif all(
                                    [
                                        x == "irrelevant" or x == "match_other_answer"
                                        for x in answer_types
                                    ]
                                ):
                                    if element not in questions_non_correct[category]:
                                        questions_non_correct[category][element] = {}
                                    if (
                                        attribute
                                        not in questions_non_correct[category][element]
                                    ):
                                        questions_non_correct[category][element][
                                            attribute
                                        ] = {}
                                    questions_non_correct[category][element][attribute] = (
                                        questions[category][element][attribute]
                                    )
                                    n_questions += 1
                                    assert (
                                        "match_other_answer" in answer_types
                                    ), "No other answer matched in question types"
                                else:
                                    raise AssertionError(
                                        f"Not covering case for {element} -- {attribute}: '{answer_types}'"
                                    )

    print("Questions that not correct: ", n_questions)
    dump_json(os.path.join(results_folder, f"qa_not_correct.json"), questions_non_correct)

def create_stats_summary(
    stats: Dict[str, Dict[str, dict]],
    category: str,
    results_folder: str
) -> Dict[str, Dict[str, int]]:
    """
    Create a summary of statistics based on the given stats dictionary.

    Args:
        stats (Dict[str, Dict[str, dict]]): A dictionary containing statistics for different question types and answer types.

    Returns:
        Dict[str, Dict[str, int]]: A summary of statistics, where the keys are question types and the values are dictionaries
        containing answer types and their corresponding counts.
    """
    del stats['generic']

    stats_summary = {qt: {at: 0 for at in stats[qt]} for qt in stats}
    total_values = {at: 0 for at in stats[next(iter(stats))]}

    num_questions = {
        qt: sum(len(stats[qt][at]) for at in stats[qt]) for qt in stats
    }
    
    assert len(set(num_questions.values())) == 1, "Not all values in the dictionary are the same."

    for qt, answers in stats.items():
        for at, items in answers.items():
            stats_summary[qt][at] = {"percent": round(len(items) / num_questions[qt] * 100, 2), "num": len(items)}
            total_values[at] += len(items)
    
    sum_val = sum(total_values[at] for at in total_values.keys())
    stats_summary["average"] = {at: 
                                    {"percent": round(total_values[at]/sum_val * 100, 2), 
                                     "num": total_values[at]}
                                 for at in total_values.keys()
                            }

    domain_dir = os.path.join(results_folder, category)
    dump_json(os.path.join(domain_dir, "scores.json"), stats_summary)
    return stats_summary

def evaluation(results_folder: str, questions_path: str):
    
    analysis = load_json(os.path.join(results_folder, "analysis.json"))
    
    total_stats_summary = {
        category: create_stats_summary(stats, category, results_folder)["average"] 
        for category, stats in analysis.items()
    }

    total_stats_summary["average"] = {
        at: {
            "num": sum(stats.get(at).get('num') for stats in total_stats_summary.values()),
            "percent": 0  
        }
        for at in ['match_correct_answer', 'match_other_answer', 'irrelevant']
    }

    total_sum = sum(entry["num"] for entry in total_stats_summary["average"].values())
    for at in total_stats_summary["average"]:
        total_stats_summary["average"][at]["percent"] = round(
            total_stats_summary["average"][at]["num"] / total_sum * 100, 2
        )
    total_stats_summary = {'average': total_stats_summary.pop('average'), **total_stats_summary}

    dump_json(os.path.join(results_folder, "scores.json"), total_stats_summary)

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m get_outdated_questions",
        description="Create the answer sheet and extract the outdated questions for a given model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "results_dir",
        metavar="DIR_NAME",
        type=str,
        help="Folder containing the generated answers from a model.",
    )
    parser.add_argument(
        "--question-path",
        metavar="FILE_PATH",
        type=str,
        default="temporal_awareness/Cognition/temporal_interval_qa.json",
        help="Path to the file containing Q&A.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    args = get_args()
    save_questions_to_update(args.results_dir, args.question_path)
    evaluation(args.results_dir, args.question_path)
