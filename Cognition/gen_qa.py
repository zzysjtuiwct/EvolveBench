import random,ipdb,sys
import argparse
import numpy as np
import torch
import spacy
from spacy.tokenizer import Tokenizer
sys.path.append('/path/project/models_output')
from utils import EXCEPTIONS, load_json, dump_json
from tqdm import tqdm
from datetime import datetime
from temporal_awareness.Awareness.analyze_replies_up2date_context_w_outdate_date import prepare_answers

football_players = ["Cristiano Ronaldo", "Lionel Messi", "Neymar Jr.","Kylian Mbapp\u00e9", "Karim Benzema", "Erling Haaland", "Mohamed Salah", "Sadio Man\u00e9", "Kevin De Bruyne", "Harry Kane"]

basketball_players = ["Stephen Curry", "Kevin Durant", "LeBron James", "Nikola Jokic", "Bradley Beal", "Giannis Antetokounmpo", "Damian Lillard", "Kawhi Leonard", "Paul George"]

F1_drivers = ["Max Verstappen", "Lewis Hamilton", "Fernando Alonso", "Sergio P\u00e9rez", "Charles Leclerc", "Lando Norris", "Carlos Sainz Jr.", "George Russell", "Pierre Gasly"]

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
nlp.tokenizer = Tokenizer(nlp.vocab)

def extract_date(date_str: str) -> str:
    """
    Extract the date part from an ISO 8601 formatted string.

    Args:
        date_str (str): Date string in the format '+YYYY-MM-DDTHH:MM:SSZ'.

    Returns:
        str: Extracted date in the format 'YYYY-MM-DD'.
    """
    return date_str.split('T')[0].lstrip('+')

def gen_format_date(date: str) -> str:
    """
    Generate a random date between two dates (inclusive).

    Args:
        start (str): Start date in the format '+YYYY-MM-DDTHH:MM:SSZ'.
        end (str): End date in the format '+YYYY-MM-DDTHH:MM:SSZ'.

    Returns:
        str: Random date in the format 'D Month YYYY'.
    """
    # Extract dates
    date_str = extract_date(date)

    # Parse extracted strings into datetime objects
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    return date_obj.strftime('%-d %B %Y')

def check_the_requirement(company):
    # 使用 spaCy 解析每个公司名称
    doc = nlp(company)
    # 检查是否包含描述性词语或已经包含 "the"
    has_the = any(token.text.lower() == "the" for token in doc)
    requires_the = any(
        token.text.lower() in {"company", "corporation", "group", "association", "organization"}
        for token in doc
    )
    
    # 判断逻辑
    if has_the:
        return False
    elif requires_the:
        return True
    else:
        return False

def rewrite_country_organization(title, entity_name, date):
    template1 = f"What is the name of the {title.lower()} of {entity_name} from {date['start']} to {date['end']}?"
    template2 = f"Who served as the {title.lower()} of {entity_name} from {date['start']} to {date['end']}?"
    template3 = f"From {date['start']} to {date['end']}, who was the {title.lower()} of {entity_name}?"

    return template1, template2, template3

def rewrite_athletes(League_name, player_name, date):
    drive_or_play = "drive" if League_name == "Formula 1 team" else "play"

    drive_or_play_t1 = "drove" if League_name == "Formula 1 team" else "played"

    template1 = f"What is the name of the {League_name} {player_name} {drive_or_play_t1} for from {date['start']} to {date['end']}?"
    template2 = f"Which {League_name} does {player_name} {drive_or_play} for from {date['start']} to {date['end']}?"
    template3 = f"From {date['start']} to {date['end']}, which {League_name} does {player_name} {drive_or_play} for?"

    return template1, template2, template3

def rewrite_Company(company_name, date):
    template1 = f"What is the name of the Chief Executive Officer of {company_name} from {date['start']} to {date['end']}?"
    template2 = f"Who served as the Chief Executive Officer of the {company_name} from {date['start']} to {date['end']}?"
    template3 = f"From {date['start']} to {date['end']}, who was the Chief Executive Officer of the {company_name}?"

    return template1, template2, template3

def main():

    parser = argparse.ArgumentParser(
        description="Generate temporal interval date for each entity.",
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
        "--temporal-interval-path",
        metavar="IMPORTANT_FILE_PATH_THAT_SAME_AS_TIME_TRAVEL_DATA",
        type=str,
        default="/path/project/models_output/temporal_awareness/Awareness/time_travel_data.json",
        help="Path to the file containing Q&A.",
    )

    args = parser.parse_args()

    original_qa = load_json(args.grc_path)
    category = list(original_qa.keys())

    temporal_interval = load_json(args.temporal_interval_path)
    
    data = {item: prepare_answers(item, original_qa, EXCEPTIONS) for item in category}

    item = {}
    for category, elements in tqdm(data.items(), desc="Categories"):
        item[category] = {}
        for element, attributes in tqdm(elements.items(), desc=f"Generating questions. for {category}"):
            item[category][element] = {}
            if category in ["countries_byGDP", "organizations"]:
                for attribute, grc_elem in attributes.items():
                    item[category][element][attribute] = {}

                    temporal_interval[category][element][attribute]['past_ground_truth']['dates']['start'] = gen_format_date(temporal_interval[category][element][attribute]['past_ground_truth']['dates']['start'])
                    temporal_interval[category][element][attribute]['past_ground_truth']['dates']['end'] = gen_format_date(temporal_interval[category][element][attribute]['past_ground_truth']['dates']['end'])

                    if category == "countries_byGDP":
                        doc = nlp(attribute)
                        for token in doc:
                            if token.dep_ == "ROOT":  # 找到核心词
                                phrase_tokens = [token.text]  # 添加核心词
                                for child in token.children:
                                    if child.dep_ in {"amod", "compound"}:  # 查找修饰词
                                        phrase_tokens.insert(0, child.text)  # 插入修饰词
                                title = " ".join(phrase_tokens)
                            if token.dep_ == "pobj" and token.head.text == "of":
                                # 使用 token.subtree 提取宾语及其修饰语
                                phrase = " ".join([child.text for child in token.subtree])

                        rw1,rw2,rw3 = rewrite_country_organization(
                                title, 
                                phrase, 
                                temporal_interval[category][element][attribute]['past_ground_truth']['dates']
                            )
                    else:
                        rw1,rw2,rw3 = rewrite_country_organization(
                                attribute, 
                                "the " + element, 
                                temporal_interval[category][element][attribute]['past_ground_truth']['dates']
                            )
                    item[category][element][attribute].update({
                                "questions":{
                                    "generic": rw1,
                                    "rephrased_1": rw1,
                                    "rephrased_2": rw2,
                                    "rephrased_3": rw3
                                },
                                "answers": original_qa[category][element][attribute]["answers"],
                                "ground_truth": temporal_interval[category][element][attribute]['past_ground_truth']['name']

                            }
                        )
            else:
                temporal_interval[category][element]['past_ground_truth']['dates']['start'] = gen_format_date(temporal_interval[category][element]['past_ground_truth']['dates']['start'])
                temporal_interval[category][element]['past_ground_truth']['dates']['end'] = gen_format_date(temporal_interval[category][element]['past_ground_truth']['dates']['end'])

                if category == "athletes_byPayment":
                    if element in football_players:
                        League_name = "football club"
                    elif element in basketball_players:
                        League_name = "basketball team"
                    elif element in F1_drivers:
                        League_name = "Formula 1 team"
                    else:
                        raise ValueError(f"Unknown athlete: {element}")
    
                    rw1,rw2,rw3 = rewrite_athletes(
                            League_name, 
                            element, 
                            temporal_interval[category][element]['past_ground_truth']['dates']
                        )
                else:
                    entity_name = "the " + element if check_the_requirement(element) else element
                    rw1,rw2,rw3 = rewrite_Company(
                            entity_name, 
                            temporal_interval[category][element]['past_ground_truth']['dates']
                        )
                item[category][element].update({
                            "questions":{
                                "generic": rw1,
                                "rephrased_1": rw1,
                                "rephrased_2": rw2,
                                "rephrased_3": rw3
                            },
                            "answers": original_qa[category][element]["answers"],
                            "ground_truth": temporal_interval[category][element]['past_ground_truth']['name']
                        }
                    )

    dump_json('temporal_interval_qa.json', item)      

if __name__ == "__main__":
    main()