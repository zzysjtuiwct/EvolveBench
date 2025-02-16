import random,ipdb,sys
import argparse
import numpy as np
import torch
import spacy
from spacy.tokenizer import Tokenizer
sys.path.append('/remote-home/zhiyuanzhu/project/DyKnow/models_output')
from utils import EXCEPTIONS, load_json, dump_json
from tqdm import tqdm
from datetime import datetime
from temporal_awareness.Reasoning.analyze_replies_reasoning import prepare_answers

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

# def rewrite_country_organization(title, entity_name, data):
#     previous_date = data["previous_event"]["in_service_date"]
#     latter_date = data["latter_event"]["in_service_date"]

#     previous_name = data["previous_event"]["name"]
#     latter_name = data["latter_event"]["name"]

#     former_or_latter = data["task_ranking"]["former_or_latter"]

#     # Ranking
#     Template_Rank_1 = f"The {title.lower()}s of {entity_name} on {previous_date} and {latter_date} were {previous_name} and {latter_name}, so who was the {former_or_latter} {title.lower()}?"

#     Template_Rank_2 = f"{previous_name} and {latter_name} served as the {title.lower()}s of {entity_name} on {previous_date} and {latter_date}, respectively. Can you identify which one the {former_or_latter} {title.lower()} was?"

#     Template_Rank_3 = f"On {previous_date}, {previous_name} was the {title.lower()} of {entity_name}, while {latter_name} held the position on {latter_date}. Which of them served as the {former_or_latter} {title.lower()}?"

#     # Accumulating
#     event, before_or_after = (data["latter_event"], "before") if data["task_accumulate"]["former_or_latter"] == "former" else (data["previous_event"], "after")
#     before_or_after_T1 = "before" if data["task_accumulate"]["former_or_latter"] == "former" else "later"

#     date = event['in_service_date']
#     name = event['name']
#     days_diff = data["task_accumulate"]["days_diff"]

#     Template_Acc_1 = f"On {date}, {name} was the {title.lower()} of {entity_name}. So, who held this position {days_diff} days {before_or_after_T1}?"

#     Template_Acc_2 = f"{name} was {entity_name}'s {title.lower()} on {date}. Do you know who was in this role {before_or_after} {days_diff} days?"

#     Template_Acc_3 = f"{name} served as the {title.lower()} of {entity_name} on {date}. Can you identify who occupied this position {before_or_after} {days_diff} days?"
    
#     return Template_Rank_1, Template_Rank_2, Template_Rank_3, Template_Acc_1, Template_Acc_2, Template_Acc_3

# def rewrite_Company(company_name, data):
#     previous_date = data["previous_event"]["in_service_date"]
#     latter_date = data["latter_event"]["in_service_date"]

#     previous_name = data["previous_event"]["name"]
#     latter_name = data["latter_event"]["name"]

#     former_or_latter = data["task_ranking"]["former_or_latter"]

#     # Ranking
#     Template_Rank_1 = f"The Chief Executive Officers of {company_name} on {previous_date} and {latter_date} were {previous_name} and {latter_name}, so who was the {former_or_latter} Chief Executive Officer?"

#     Template_Rank_2 = f"{previous_name} and {latter_name} served as the Chief Executive Officers of {company_name} on {previous_date} and {latter_date}, respectively. Can you identify which one the {former_or_latter} Chief Executive Officer was?"

#     Template_Rank_3 = f"On {previous_date}, {previous_name} was the Chief Executive Officer of {company_name}, while {latter_name} held the position on {latter_date}. Which of them served as the {former_or_latter} Chief Executive Officer?"

#     # Accumulating
#     event, before_or_after = (data["latter_event"], "before") if data["task_accumulate"]["former_or_latter"] == "former" else (data["previous_event"], "after")
#     before_or_after_T1 = "before" if data["task_accumulate"]["former_or_latter"] == "former" else "later"

#     date = event['in_service_date']
#     name = event['name']
#     days_diff = data["task_accumulate"]["days_diff"]

#     Template_Acc_1 = f"On {date}, {name} was the Chief Executive Officer of {company_name}. So, who held this position {days_diff} days {before_or_after_T1}?"

#     Template_Acc_2 = f"{name} was {company_name}'s Chief Executive Officer on {date}. Do you know who was in this role {before_or_after} {days_diff} days?"

#     Template_Acc_3 = f"{name} served as the Chief Executive Officer of {company_name} on {date}. Can you identify who occupied this position {before_or_after} {days_diff} days?"
    
#     return Template_Rank_1, Template_Rank_2, Template_Rank_3, Template_Acc_1, Template_Acc_2, Template_Acc_3

# def rewrite_athletes(League_name, player_name, data):
#     drive_or_play = "drive" if League_name == "Formula 1 team" else "play"
#     drive_or_play_t1 = "drove" if League_name == "Formula 1 team" else "played"

#     previous_date = data["previous_event"]["in_service_date"]
#     latter_date = data["latter_event"]["in_service_date"]

#     previous_name = data["previous_event"]["name"]
#     latter_name = data["latter_event"]["name"]

#     former_or_latter = data["task_ranking"]["former_or_latter"]

#     # Ranking
#     Template_Rank_1 = f"The {League_name}s of {player_name} on {previous_date} and {latter_date} were {previous_name} and {latter_name}, so which team was the {former_or_latter} one {player_name} {drive_or_play_t1} for?"

#     Template_Rank_2 = f"{previous_name} and {latter_name} were {player_name}'s {League_name}s on {previous_date} and {latter_date}, respectively. Can you identify which team was the {former_or_latter} one {player_name} {drive_or_play_t1} for?"

#     Template_Rank_3 = f"On {previous_date}, {player_name} {drive_or_play_t1} for {previous_name}, while {player_name} {drive_or_play_t1} for {latter_name} on {latter_date}. Which team did {player_name} {former_or_latter}ly {drive_or_play} for?"

#     # Accumulating
#     event, before_or_after = (data["latter_event"], "before") if data["task_accumulate"]["former_or_latter"] == "former" else (data["previous_event"], "after")

#     date = event['in_service_date']
#     name = event['name']
#     days_diff = data["task_accumulate"]["days_diff"]

#     Template_Acc_1 = f"On {date}, {player_name} {drive_or_play_t1} for {name}. So, {before_or_after} {days_diff} days, which {League_name} did {player_name} {drive_or_play} for?"

#     Template_Acc_2 = f"{name} was {player_name}'s {League_name} on {date}. Can you identify which {League_name} {player_name} belongs to {before_or_after} {days_diff} days?"

#     Template_Acc_3 = f"{player_name} {drive_or_play_t1} for {name} on {date}. Do you know which {League_name} {player_name} {drive_or_play_t1} for {before_or_after} {days_diff} days?"

#     return Template_Rank_1, Template_Rank_2, Template_Rank_3, Template_Acc_1, Template_Acc_2, Template_Acc_3
    
def rewrite_country_organization_notime(title, entity_name, data):
    previous_date = data["previous_event"]["in_service_date"]
    latter_date = data["latter_event"]["in_service_date"]

    previous_name = data["previous_event"]["name"]
    latter_name = data["latter_event"]["name"]

    former_or_latter = data["task_ranking"]["former_or_latter"]

    # Ranking
    Template_Rank_1 = f"{previous_name} and {latter_name} both served as the {title.lower()}s of {entity_name} before. Do you know which one of them the {former_or_latter} {title.lower()} was?"

    Template_Rank_2 = f"{entity_name} had {previous_name} and {latter_name} as former {title.lower()}s. Can you identify which one the {former_or_latter} {title.lower()} was?"

    Template_Rank_3 = f"{previous_name} and {latter_name} previously held the position of the {title.lower()} of {entity_name}. Which of them served as the {former_or_latter} {title.lower()}?"

    # Accumulating
    event, before_or_after = (data["latter_event"], "before") if data["task_accumulate"]["former_or_latter"] == "former" else (data["previous_event"], "after")
    before_or_after_T1 = "before" if data["task_accumulate"]["former_or_latter"] == "former" else "later"

    date = event['in_service_date']
    name = event['name']
    days_diff = data["task_accumulate"]["days_diff"]

    days2year = round(days_diff / 365.25)

    Template_Acc_1 = f"On {date}, {name} was the {title.lower()} of {entity_name}. So, who held this position {days2year} years {before_or_after_T1}?"

    Template_Acc_2 = f"{name} was {entity_name}'s {title.lower()} on {date}. Do you know who was in this role {before_or_after} {days2year} years?"

    Template_Acc_3 = f"{name} served as the {title.lower()} of {entity_name} on {date}. Can you identify who occupied this position {before_or_after} {days2year} years?"
    
    return Template_Rank_1, Template_Rank_2, Template_Rank_3, Template_Acc_1, Template_Acc_2, Template_Acc_3

def rewrite_Company_notime(company_name, data):
    previous_date = data["previous_event"]["in_service_date"]
    latter_date = data["latter_event"]["in_service_date"]

    previous_name = data["previous_event"]["name"]
    latter_name = data["latter_event"]["name"]

    former_or_latter = data["task_ranking"]["former_or_latter"]

    # Ranking
    Template_Rank_1 = f"{previous_name} and {latter_name} both served as Chief Executive Officers of {company_name} before. Do you know which one of them the {former_or_latter} Chief Executive Officer was?"

    Template_Rank_2 = f"{company_name} had {previous_name} and {latter_name} as former Chief Executive Officers. Can you identify which one of them was the {former_or_latter} Chief Executive Officer?"

    Template_Rank_3 = f"{previous_name} and {latter_name} previously held the position of {company_name}'s Chief Executive Officer. Which of them served as the {former_or_latter} Chief Executive Officer?"

    # Accumulating
    event, before_or_after = (data["latter_event"], "before") if data["task_accumulate"]["former_or_latter"] == "former" else (data["previous_event"], "after")
    before_or_after_T1 = "before" if data["task_accumulate"]["former_or_latter"] == "former" else "later"

    date = event['in_service_date']
    name = event['name']
    days_diff = data["task_accumulate"]["days_diff"]

    days2year = round(days_diff / 365.25)

    Template_Acc_1 = f"On {date}, {name} was the Chief Executive Officer of {company_name}. So, who held this position {days2year} years {before_or_after_T1}?"

    Template_Acc_2 = f"{name} was {company_name}'s Chief Executive Officer on {date}. Do you know who was in this role {before_or_after} {days2year} years?"

    Template_Acc_3 = f"{name} served as the Chief Executive Officer of {company_name} on {date}. Can you identify who occupied this position {before_or_after} {days2year} years?"
    
    return Template_Rank_1, Template_Rank_2, Template_Rank_3, Template_Acc_1, Template_Acc_2, Template_Acc_3

def rewrite_athletes_notime(League_name, player_name, data):
    drive_or_play = "drive" if League_name == "Formula 1 team" else "play"
    drive_or_play_t1 = "drove" if League_name == "Formula 1 team" else "played"

    previous_date = data["previous_event"]["in_service_date"]
    latter_date = data["latter_event"]["in_service_date"]

    previous_name = data["previous_event"]["name"]
    latter_name = data["latter_event"]["name"]

    former_or_latter = data["task_ranking"]["former_or_latter"]

    # Ranking
    Template_Rank_1 = f"{previous_name} and {latter_name} were both {player_name}'s {League_name}s before, so which team between them was the {former_or_latter} one {player_name} {drive_or_play_t1} for?"

    Template_Rank_2 = f"{player_name} has joined {previous_name} and {latter_name} {League_name}s before. Can you identify which team between them was the {former_or_latter} one {player_name} {drive_or_play_t1} for?"

    Template_Rank_3 = f"{player_name} previously {drive_or_play_t1} for {previous_name} and {latter_name} {League_name}s. Which team between them did {player_name} {former_or_latter}ly {drive_or_play} for?"
    # Accumulating
    event, before_or_after = (data["latter_event"], "before") if data["task_accumulate"]["former_or_latter"] == "former" else (data["previous_event"], "after")

    date = event['in_service_date']
    name = event['name']
    days_diff = data["task_accumulate"]["days_diff"]

    days2year = round(days_diff / 365.25)

    Template_Acc_1 = f"On {date}, {player_name} {drive_or_play_t1} for {name}. So, {before_or_after} {days2year} years, which {League_name} did {player_name} {drive_or_play} for?"

    Template_Acc_2 = f"{name} was {player_name}'s {League_name} on {date}. Can you identify which {League_name} {player_name} belongs to {before_or_after} {days2year} years?"

    Template_Acc_3 = f"{player_name} {drive_or_play_t1} for {name} on {date}. Do you know which {League_name} {player_name} {drive_or_play_t1} for {before_or_after} {days2year} years?"

    return Template_Rank_1, Template_Rank_2, Template_Rank_3, Template_Acc_1, Template_Acc_2, Template_Acc_3

def main():

    parser = argparse.ArgumentParser(
        description="Generate temporal reasoning date.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grc-path",
        metavar="FILE_PATH",
        type=str,
        default="/remote-home/zhiyuanzhu/project/DyKnow/my_data/question/up2dated_qa.json",
        help="Path to the file containing Q&A.",
    )

    parser.add_argument(
        "--temporal_reasoning_task",
        metavar="IMPORTANT_FILE_PATH_THAT_SAME_AS_TIME_TRAVEL_DATA",
        type=str,
        default="reasoning_task_data.json",
        help="Path to the file containing Q&A.",
    )

    args = parser.parse_args()

    original_qa = load_json(args.grc_path)
    category = list(original_qa.keys())

    reasoning_data = load_json(args.temporal_reasoning_task)
    
    data = {cate: prepare_answers(cate, original_qa, EXCEPTIONS) for cate in category}

    item = {}
    for category, elements in tqdm(data.items(), desc="Categories"):
        item[category] = {}
        for element, attributes in tqdm(elements.items(), desc=f"Generating questions. for {category}"):
            item[category][element] = {}
            if category in ["countries_byGDP", "organizations"]:
                for attribute, grc_elem in attributes.items():
                    item[category][element][attribute] = {}

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

                        R_rw1,R_rw2,R_rw3,T_rw1,T_rw2,T_rw3 = rewrite_country_organization_notime(
                                title, 
                                phrase, 
                                reasoning_data[category][element][attribute]
                            )
                    else:
                        R_rw1,R_rw2,R_rw3,T_rw1,T_rw2,T_rw3 = rewrite_country_organization_notime(
                                attribute, 
                                "the " + element, 
                                reasoning_data[category][element][attribute]
                            )
                    
                    assert reasoning_data[category][element][attribute]['task_ranking']['ground_truth'] == reasoning_data[category][element][attribute]['task_accumulate']['ground_truth']

                    item[category][element][attribute].update({
                                "answers": original_qa[category][element][attribute]["answers"],
                                "ground_truth": reasoning_data[category][element][attribute]['task_ranking']['ground_truth'],
                                "ranking_qa":{
                                    "generic": R_rw1,
                                    "rephrased_1": R_rw1,
                                    "rephrased_2": R_rw2,
                                    "rephrased_3": R_rw3,
                                },
                                "accumulate_qa":{
                                    "generic": T_rw1,
                                    "rephrased_1": T_rw1,
                                    "rephrased_2": T_rw2,
                                    "rephrased_3": T_rw3,
                                },
                            }
                        )
            else:
                if category == "athletes_byPayment":
                    if element in football_players:
                        League_name = "football club"
                    elif element in basketball_players:
                        League_name = "basketball team"
                    elif element in F1_drivers:
                        League_name = "Formula 1 team"
                    else:
                        raise ValueError(f"Unknown athlete: {element}")
    
                    R_rw1,R_rw2,R_rw3,T_rw1,T_rw2,T_rw3 = rewrite_athletes_notime(
                            League_name, 
                            element, 
                            reasoning_data[category][element]
                        )
                else:
                    entity_name = "the " + element if check_the_requirement(element) else element
                    R_rw1,R_rw2,R_rw3,T_rw1,T_rw2,T_rw3 = rewrite_Company_notime(
                            entity_name, 
                            reasoning_data[category][element]
                        )
                    
                assert reasoning_data[category][element]['task_ranking']['ground_truth'] == reasoning_data[category][element]['task_accumulate']['ground_truth']
                    
                item[category][element].update({
                            "answers": original_qa[category][element]["answers"],
                            "ground_truth": reasoning_data[category][element]['task_ranking']['ground_truth'],
                            "ranking_qa":{
                                "generic": R_rw1,
                                "rephrased_1": R_rw1,
                                "rephrased_2": R_rw2,
                                "rephrased_3": R_rw3,
                            },
                            "accumulate_qa":{
                                "generic": T_rw1,
                                "rephrased_1": T_rw1,
                                "rephrased_2": T_rw2,
                                "rephrased_3": T_rw3,
                            },
                        }
                    )

    dump_json('reasoning_qa_20250210.json', item)      

if __name__ == "__main__":
    main()