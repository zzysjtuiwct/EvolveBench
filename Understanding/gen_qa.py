import re,json,ipdb
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
import spacy
from spacy.tokenizer import Tokenizer
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from refine_data import load_json, dump_json, prepare_time_event, prepare_answers, EXCEPTIONS

football_players = ["Cristiano Ronaldo", "Lionel Messi", "Neymar Jr.","Kylian Mbapp\u00e9", "Karim Benzema", "Erling Haaland", "Mohamed Salah", "Sadio Man\u00e9", "Kevin De Bruyne", "Harry Kane"]

basketball_players = ["Stephen Curry", "Kevin Durant", "LeBron James", "Nikola Jokic", "Bradley Beal", "Giannis Antetokounmpo", "Damian Lillard", "Kawhi Leonard", "Paul George"]

F1_drivers = ["Max Verstappen", "Lewis Hamilton", "Fernando Alonso", "Sergio P\u00e9rez", "Charles Leclerc", "Lando Norris", "Carlos Sainz Jr.", "George Russell", "Pierre Gasly"]


spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
nlp.tokenizer = Tokenizer(nlp.vocab)

def gen_format_date(date: str) -> str:
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    return date_obj.strftime('%-d %B %Y')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def rewrite_country_organization(event, title, entity_name):
    co_event_name = event['item']['name']
    att = event['instance']
    description = f"{co_event_name} was the {att}"

    implict_template1 = f"What was the name of the {title.lower()} of {entity_name} when {description}?"
    implict_template2 = f"Who served as the {title.lower()} of {entity_name} when {description}?"
    implict_template3 = f"When {description}, who was the {title.lower()} of {entity_name}?"

    date_start = gen_format_date(event['item']['start'])
    date_end = gen_format_date(event['item']['end'])
    explict_template1 = f"What was the name of the {title.lower()} of {entity_name} from {date_start} to {date_end}?"
    explict_template2 = f"Who served as the {title.lower()} of {entity_name} from {date_start} to {date_end}?"
    explict_template3 = f"From {date_start} to {date_end}, who was the {title.lower()} of {entity_name}?"
    
    return implict_template1, implict_template2, implict_template3, explict_template1, explict_template2, explict_template3

def rewrite_athletes(event, League_name, player_name):
    co_event_name = event['item']['name']
    att = event['instance']
    description = f"{co_event_name} was the {att}"

    drive_or_play = "drive" if League_name == "Formula 1 team" else "play"
    drive_or_play_t1 = "drove" if League_name == "Formula 1 team" else "played"

    implict_template1 = f"What was the name of the {League_name} {player_name} {drive_or_play_t1} for when {description}?"
    implict_template2 = f"Which {League_name} did {player_name} {drive_or_play} for when {description}?"
    implict_template3 = f"When {description}, which {League_name} did {player_name} {drive_or_play} for?"

    date_start = gen_format_date(event['item']['start'])
    date_end = gen_format_date(event['item']['end'])
    explict_template1 = f"What was the name of the {League_name} {player_name} {drive_or_play_t1} for from {date_start} to {date_end}?"
    explict_template2 = f"Which {League_name} did {player_name} {drive_or_play} for from {date_start} to {date_end}?"
    explict_template3 = f"From {date_start} to {date_end}, which {League_name} did {player_name} {drive_or_play} for?"

    return implict_template1, implict_template2, implict_template3, explict_template1, explict_template2, explict_template3

def rewrite_Company(event, company_name):
    co_event_name = event['item']['name']
    att = event['instance']
    description = f"{co_event_name} was the {att}"

    implict_template1 = f"What was the name of the Chief Executive Officer of {company_name} when {description}?"
    implict_template2 = f"Who served as the Chief Executive Officer of {company_name} when {description}?"
    implict_template3 = f"When {description}, who was the Chief Executive Officer of {company_name}?"

    date_start = gen_format_date(event['item']['start'])
    date_end = gen_format_date(event['item']['end'])
    explict_template1 = f"What was the name of the Chief Executive Officer of {company_name} from {date_start} to {date_end}?"
    explict_template2 = f"Who served as the Chief Executive Officer of {company_name} from {date_start} to {date_end}?"
    explict_template3 = f"From {date_start} to {date_end}, who was the Chief Executive Officer of {company_name}?"

    return implict_template1, implict_template2, implict_template3, explict_template1, explict_template2, explict_template3


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
    
def main():
    set_seed(30)

    original_qa = load_json("/remote-home/zhiyuanzhu/project/DyKnow/my_data/question/up2dated_qa.json")
    original_answer = {item: prepare_answers(item, original_qa, EXCEPTIONS) for item in original_qa.keys()}

    event_data = load_json("country_event.json")

    data_to_analyze = {}
    for domain in original_qa:
        if domain not in data_to_analyze:
            data_to_analyze[domain] = {}
        for element in tqdm(original_qa[domain], desc=domain):
            if element not in data_to_analyze[domain]:
                data_to_analyze[domain][element] = {}
            if domain in ["countries_byGDP", "organizations"]:
                for attribute in original_qa[domain][element]:
                    if attribute not in data_to_analyze[domain][element]:
                        data_to_analyze[domain][element][attribute] = {}
                    
                    answer = original_answer[domain][element][attribute]
                    while True:
                        key = random.choice(list(event_data[domain][element][attribute]["answers"].keys()))
                        candi_name, candi_start_time, candi_end_time = prepare_time_event(key)
                        candi_start_time = re.sub("-00", "-01", candi_start_time.strip()) if candi_start_time else None
                        candi_end_time = re.sub("-00", "-01", candi_end_time.strip()) if candi_end_time else None
                        if candi_name in answer:
                            if candi_start_time == answer[candi_name]['start'] and candi_end_time == answer[candi_name]['end']:
                                co_event = random.choice(event_data[domain][element][attribute]["answers"][key])
                                co_event_name, co_event_start_time, co_event_end_time = prepare_time_event(co_event['item'])
                                if co_event_end_time is not None:
                                    break
    
                    co_event['item'] = {"name": co_event_name, "start": co_event_start_time, "end": co_event_end_time}

                    if domain == "countries_byGDP":
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

                        im_rw1,im_rw2,im_rw3, exp_rw1, exp_rw2, exp_rw3 = rewrite_country_organization(co_event, title, phrase)
                    else:
                        im_rw1,im_rw2,im_rw3, exp_rw1, exp_rw2, exp_rw3 = rewrite_country_organization(
                                co_event,
                                attribute, 
                                "the " + element
                            )
                    data_to_analyze[domain][element][attribute].update({
                        "implict": {
                            "generic": im_rw1,
                            "rephrased_1": im_rw1,
                            "rephrased_2": im_rw2,
                            "rephrased_3": im_rw3
                        },
                        "explict": {
                            "generic": exp_rw1,
                            "rephrased_1": exp_rw1,
                            "rephrased_2": exp_rw2,
                            "rephrased_3": exp_rw3
                        },
                        "answers": original_qa[domain][element][attribute]["answers"],
                        "ground_truth": {
                            "name": candi_name,
                            "start": candi_start_time,
                            "end": candi_end_time
                        },
                        "co_event": {
                            "name": co_event_name,
                            "start": co_event_start_time,
                            "end": co_event_end_time,
                            "element": co_event['entity'],
                            "attribute": co_event['instance'],
                            "description": co_event['description']
                        }
                    })
            else:
                answer = original_answer[domain][element]
                while True:
                    key = random.choice(list(event_data[domain][element]["answers"].keys()))
                    candi_name, candi_start_time, candi_end_time = prepare_time_event(key)
                    candi_start_time = re.sub("-00", "-01", candi_start_time.strip()) if candi_start_time else None
                    candi_end_time = re.sub("-00", "-01", candi_end_time.strip()) if candi_end_time else None
                    if candi_name in answer:
                        if candi_start_time == answer[candi_name]['start'] and candi_end_time == answer[candi_name]['end']:
                            co_event = random.choice(event_data[domain][element]["answers"][key])
                            co_event_name, co_event_start_time, co_event_end_time = prepare_time_event(co_event['item'])
                            if co_event_end_time is not None:
                                break
                    
                co_event['item'] = {"name": co_event_name, "start": co_event_start_time, "end": co_event_end_time}

                if domain == "athletes_byPayment":
                    if element in football_players:
                        League_name = "football club"
                    elif element in basketball_players:
                        League_name = "basketball team"
                    elif element in F1_drivers:
                        League_name = "Formula 1 team"
                    else:
                        raise ValueError(f"Unknown athlete: {element}")
    
                    im_rw1,im_rw2,im_rw3, exp_rw1, exp_rw2, exp_rw3 = rewrite_athletes(
                            co_event,
                            League_name, 
                            element
                        )
                else:
                    entity_name = "the " + element if check_the_requirement(element) else element
                    im_rw1,im_rw2,im_rw3, exp_rw1, exp_rw2, exp_rw3 = rewrite_Company(
                            co_event,
                            entity_name
                        )
                        
                data_to_analyze[domain][element].update({
                    "implict": {
                        "generic": im_rw1,
                        "rephrased_1": im_rw1,
                        "rephrased_2": im_rw2,
                        "rephrased_3": im_rw3
                    },
                    "explict": {
                        "generic": exp_rw1,
                        "rephrased_1": exp_rw1,
                        "rephrased_2": exp_rw2,
                        "rephrased_3": exp_rw3
                    },
                    "answers": original_qa[domain][element]["answers"],
                    "ground_truth": {
                        "name": candi_name,
                        "start": candi_start_time,
                        "end": candi_end_time
                    },
                    "co_event": {
                        "name": co_event_name,
                        "start": co_event_start_time,
                        "end": co_event_end_time,
                        "element": co_event['entity'],
                        "attribute": co_event['instance'],
                        "description": co_event['description']
                    }
                })

    dump_json('implict_qa_.json', data_to_analyze)      


if __name__ == "__main__":
    main()