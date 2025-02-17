import random,ipdb,sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
sys.path.append('/path/project/models_output')
from utils import EXCEPTIONS, load_json, dump_json
from analyze_replies import prepare_answers

def extract_date(date_str: str) -> str:
    """
    Extract the date part from an ISO 8601 formatted string.

    Args:
        date_str (str): Date string in the format '+YYYY-MM-DDTHH:MM:SSZ'.

    Returns:
        str: Extracted date in the format 'YYYY-MM-DD'.
    """
    return date_str.split('T')[0].lstrip('+')

def format_date(date_str: str):
    return datetime.strptime(extract_date(date_str), '%Y-%m-%d')

def random_date(start: str, end: str) -> str:
    """
    Generate a random date between two dates (inclusive).

    Args:
        start (str): Start date in the format '+YYYY-MM-DDTHH:MM:SSZ'.
        end (str): End date in the format '+YYYY-MM-DDTHH:MM:SSZ'.

    Returns:
        str: Random date in the format 'D Month YYYY'.
    """
    # Extract dates
    start_date_str = extract_date(start)
    end_date_str = extract_date(end)

    # Parse extracted strings into datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Ensure the start date is not after the end date
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date.")

    # Calculate the range of days between start and end
    delta_days = (end_date - start_date).days

    # Generate a random number of days to add to the start date
    random_days = random.randint(0, delta_days)

    # Calculate the random date
    random_date = start_date + timedelta(days=random_days)

    return random_date.strftime('%-d %B %Y')

def get_service_date(item):
    end_date = item[-1]['end'] if item[-1]['end'] else "+2025-01-01T00:00:00Z"
    return random_date(item[-1]['start'], end_date)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(10000)

    parser = argparse.ArgumentParser(
        description="Generate data for reasoning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grc-path",
        metavar="FILE_PATH",
        type=str,
        default="/path/project/my_data/question/up2dated_qa.json",
        help="Path to the file containing Q&A.",
    )

    args = parser.parse_args()

    original_qa = load_json(args.grc_path)
    category = list(original_qa.keys())

    data = {item: prepare_answers(item, original_qa, EXCEPTIONS) for item in category}

    item = {}
    for category, elements in tqdm(data.items(), desc="Categories"):
        item[category] = {}
        for element, attributes in tqdm(elements.items(), desc=f"Generating Answ. for {category}"):
            item[category][element] = {}
            if category in ["countries_byGDP", "organizations"]:
                for attribute, grc_elem in attributes.items():
                    item[category][element][attribute] = {}

                    sample_list = list(grc_elem.items())
                    if len(sample_list) <= 2:
                        assert len(sample_list) == 2
                    else:
                        idx = []
                        for id, (name, dates) in enumerate(sample_list):
                            if dates['end'] is None:
                                idx.append(id)

                        assert len(idx) == 1
                        _, _ = sample_list.pop(idx.pop())

                    while True:
                        item_1, item_2 = random.sample(sample_list, 2)
                        prev_event, late_event = (item_1, item_2) if format_date(item_1[-1]['start']) < format_date(item_2[-1]['start']) else (item_2, item_1)
                        if format_date(prev_event[-1]['end']) <= format_date(late_event[-1]['start']):
                            break

                    Date_of_service_prev = get_service_date(prev_event)
                    Date_of_service_late = get_service_date(late_event)

                    days_diff = (datetime.strptime(Date_of_service_late, "%d %B %Y") - datetime.strptime(Date_of_service_prev, "%d %B %Y")).days
                    
                    former_or_latter = random.choice(["former", "latter"])

                    item[category][element][attribute].update({
                            "previous_event": {
                                "name": prev_event[0],
                                "info": prev_event[1],
                                "in_service_date": Date_of_service_prev
                            },
                            "latter_event": {
                                "name": late_event[0],
                                "info": late_event[1],
                                "in_service_date": Date_of_service_late
                            },
                            "task_ranking":{
                                "former_or_latter": former_or_latter,
                                "ground_truth": prev_event[0] if former_or_latter == "former" else late_event[0]
                            },
                            "task_accumulate":{
                                "former_or_latter": former_or_latter,
                                "days_diff": days_diff,
                                "ground_truth": prev_event[0] if former_or_latter == "former" else late_event[0]
                            }
                        }
                    )
            else:
                sample_list = list(attributes.items())

                if len(sample_list) <= 2:
                    assert len(sample_list) == 2
                else:
                    idx = []
                    for id, (name, dates) in enumerate(sample_list):
                        if dates['end'] is None:
                            idx.append(id)

                    assert len(idx) == 1
                    _, _ = sample_list.pop(idx.pop())

                while True:
                    item_1, item_2 = random.sample(sample_list, 2)
                    prev_event, late_event = (item_1, item_2) if format_date(item_1[-1]['start']) < format_date(item_2[-1]['start']) else (item_2, item_1)
                    if format_date(prev_event[-1]['end']) <= format_date(late_event[-1]['start']):
                        break

                prev_event, late_event = (item_1, item_2) if format_date(item_1[-1]['start']) < format_date(item_2[-1]['start']) else (item_2, item_1)

                Date_of_service_prev = get_service_date(prev_event)
                Date_of_service_late = get_service_date(late_event)

                days_diff = (datetime.strptime(Date_of_service_late, "%d %B %Y") - datetime.strptime(Date_of_service_prev, "%d %B %Y")).days

                former_or_latter = random.choice(["former", "latter"])
                
                item[category][element].update({
                        "previous_event": {
                            "name": prev_event[0],
                            "info": prev_event[1],
                            "in_service_date": Date_of_service_prev
                        },
                        "latter_event": {
                            "name": late_event[0],
                            "info": late_event[1],
                            "in_service_date": Date_of_service_late
                        },
                        "task_ranking":{
                            "former_or_latter": former_or_latter,
                            "ground_truth": prev_event[0] if former_or_latter == "former" else late_event[0]
                        },
                        "task_accumulate":{
                            "former_or_latter": former_or_latter,
                            "days_diff": days_diff,
                            "ground_truth": prev_event[0] if former_or_latter == "former" else late_event[0]
                        }
                    }
                )

    dump_json('reasoning_task_data_.json', item)


if __name__ == "__main__":
    main()