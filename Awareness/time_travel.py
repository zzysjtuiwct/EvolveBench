import random,ipdb,sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
sys.path.append('/path/project/models_output')
from utils import EXCEPTIONS, load_json, dump_json
from temporal_awareness.Awareness.analyze_replies_up2date_context_w_outdate_date import prepare_answers

def extract_date(date_str: str) -> str:
    """
    Extract the date part from an ISO 8601 formatted string.

    Args:
        date_str (str): Date string in the format '+YYYY-MM-DDTHH:MM:SSZ'.

    Returns:
        str: Extracted date in the format 'YYYY-MM-DD'.
    """
    return date_str.split('T')[0].lstrip('+')

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate past date for each entity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grc-path",
        metavar="FILE_PATH",
        type=str,
        default="/path/project/my_data/question/up2dated_qa.json",
        help="Path to the file containing Q&A.",
    )
    parser.add_argument(
        "--future-newspaper",
        metavar="FILE_PATH",
        type=str,
        default="/path/project/my_data/passage/passages_no_time_info.json",
        help="Path to the file containing future context.",
    )

    args = parser.parse_args()

    up2date_context = load_json(args.future_newspaper)

    original = load_json(args.grc_path)
    category = list(original.keys())

    data = {item: prepare_answers(item, original, EXCEPTIONS) for item in category}

    item = {}
    for category, elements in tqdm(data.items(), desc="Categories"):
        item[category] = {}
        for element, attributes in tqdm(elements.items(), desc=f"Generating Answ. for {category}"):
            item[category][element] = {}
            if category in ["countries_byGDP", "organizations"]:
                for attribute, grc_elem in attributes.items():
                    item[category][element][attribute] = {}
                    '''remove the items with no end date'''
                    idx = []
                    sample_list = list(grc_elem.items())
                    for id, (name, dates) in enumerate(sample_list):
                        if dates['end'] is None:
                            idx.append(id) 

                    assert len(idx)==1
                    _, _ = sample_list.pop(idx.pop())

                    while True:
                        outdate_name, outdate_dates = sample_list[random.randint(0, len(sample_list) - 1)]
                        if extract_date(outdate_dates['start']) != extract_date(outdate_dates['end']):
                            break
                    
                    item[category][element][attribute].update({'past_ground_truth':{'name':outdate_name, 'dates':outdate_dates}})
                    item[category][element][attribute].update({'time_travel_date': random_date(outdate_dates['start'], outdate_dates['end'])})
                    item[category][element][attribute].update({'future_news': up2date_context[category][element][attribute]['matches']['up2date_knowledge']})
            else:
                '''remove the items with no end date'''
                idx = []
                sample_list = list(attributes.items())
                for id, (name, dates) in enumerate(sample_list):
                    if dates['end'] is None:
                        idx.append(id)

                assert len(idx)==1
                _, _ = sample_list.pop(idx.pop())

                while True:
                    outdate_name, outdate_dates = sample_list[random.randint(0, len(sample_list) - 1)]
                    if extract_date(outdate_dates['start']) != extract_date(outdate_dates['end']):
                        break

                item[category][element].update({'past_ground_truth':{'name':outdate_name, 'dates':outdate_dates}})
                item[category][element].update({'time_travel_date': random_date(outdate_dates['start'], outdate_dates['end'])})
                item[category][element].update({'future_news': up2date_context[category][element]['matches']['up2date_knowledge']})
    
    dump_json('time_travel_data_no_time_info.json', item)
if __name__ == "__main__":
    main()