import random,ipdb,sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
sys.path.append('/remote-home/zhiyuanzhu/project/DyKnow/models_output')
from utils import EXCEPTIONS, load_json, dump_json
from analyze_replies import prepare_answers
from temporal_awareness.Awareness.time_travel import extract_date

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
        description="Generate unanswerable date",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grc-path",
        metavar="FILE_PATH",
        type=str,
        default="/remote-home/zhiyuanzhu/project/DyKnow/my_data/question/up2dated_qa.json",
        help="Path to the file containing Q&A.",
    )

    args = parser.parse_args()

    original = load_json(args.grc_path)
    category = list(original.keys())
    
    data = {item: prepare_answers(item, original, EXCEPTIONS) for item in category}

    previous_item = load_json('/remote-home/zhiyuanzhu/project/DyKnow/models_output/temporal_awareness/Trustworthiness/unanswerable_date_human.json')
    item = {}
    for category, elements in tqdm(data.items(), desc="Categories"):
        item[category] = {}
        for element, attributes in tqdm(elements.items(), desc=f"Generating Answ. for {category}"):
            item[category][element] = {}
            if category in ["countries_byGDP", "organizations"]:
                for attribute, grc_elem in attributes.items():
                    item[category][element][attribute] = {}
                    
                    earlyist_date = datetime.strptime('2025-10-01', '%Y-%m-%d')
                    earlyist_event = ""

                    sample_list = list(grc_elem.items())
                    for id, (name, dates) in enumerate(sample_list):
                        item_start = datetime.strptime(extract_date(dates['start']), '%Y-%m-%d')
                        if item_start < earlyist_date:
                            earlyist_date = item_start
                            earlyist_event = name

                    try:
                        earlyist_date = earlyist_date.replace(year=earlyist_date.year - 10)
                    except ValueError:
                        earlyist_date = earlyist_date.replace(month=2, day=28, year=earlyist_date.year - 10)
                    if element in previous_item[category]:
                        item[category][element][attribute] = previous_item[category][element][attribute]
                    else:
                        item[category][element][attribute].update({'unanswerable_date':
                                                                    {
                                                                        'earlyist_name':earlyist_event, 
                                                                        'past':earlyist_date.strftime('%-d %B %Y'),
                                                                        'future': datetime.strptime('2050-10-01', '%Y-%m-%d').strftime('%-d %B %Y')
                                                                        }
                                                                    }
                                                                )
            else:
                earlyist_date = datetime.strptime('2025-10-01', '%Y-%m-%d')
                earlyist_event = ""

                sample_list = list(attributes.items())
                for id, (name, dates) in enumerate(sample_list):
                    item_start = datetime.strptime(extract_date(dates['start']), '%Y-%m-%d')
                    if item_start < earlyist_date:
                        earlyist_date = item_start
                        earlyist_event = name
                    
                try:
                    earlyist_date = earlyist_date.replace(year=earlyist_date.year - 10)
                except ValueError:
                    earlyist_date = earlyist_date.replace(month=2, day=28, year=earlyist_date.year - 10)
                if element in previous_item[category]:
                    item[category][element] = previous_item[category][element]
                else:
                    item[category][element].update({'unanswerable_date':
                                                    {
                                                        'earlyist_name':earlyist_event, 
                                                        'past':earlyist_date.strftime('%-d %B %Y'),
                                                        'future': datetime.strptime('2050-10-01', '%Y-%m-%d').strftime('%-d %B %Y')
                                                        }
                                                    }
                                                )
    
    dump_json('unanswerable_date_.json', item)

if __name__ == "__main__":
    main()