import re,json,ipdb
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Set, Any

EXCEPTIONS = {
    "athletes_byPayment": {
        "Lionel Messi": "Argentina national association football team",
        "Neymar Jr.": "Brazil national football team",
        "Kylian Mbappé": "France national association football team",
        "Mohamed Salah": "Egypt national football team",
        "Sadio Mané": "Senegal national association football team",
        "Kevin De Bruyne": "Belgium national football team",
        "Harry Kane": "England national association football team",
    }
}
# 这里都是只有开始时间没有结束时间的item，会导致answer列表里有多于一个end date是None的answer
# 但是像一般问梅西当前效力的球队一般都是指的是职业联赛球队，上面这些结果里都是因为一个球员同时
# 效力国家队和职业球队，所以才会导致有两个end为None的结果。（现在我应该已经把所有国家队的删掉了）
def load_json(path: str) -> Any:
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file

def dump_json(path: str, obj: Any, indent: int = 4):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)

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
    for name, dates in to_assign.items():
        dates['start'] = re.search(r"\d{4}-\d{2}-\d{2}", dates['start']).group() if dates['start'] else None
        dates['end'] = re.search(r"\d{4}-\d{2}-\d{2}", dates['end']).group() if dates['end'] else None

    return to_assign

def prepare_answers(category: str, original: dict, exceptions: dict) -> dict:
    answers = {}

    if category in ["countries_byGDP", "organizations"]:
        for element, attributes in original[category].items():
            if element not in answers:
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

def prepare_time_event(implict_time_event: dict) -> dict:
    import re

    # 原始字符串
    text = "Carlo Azeglio Ciampi |S: +1999-05-18T00:00:00Z |E: +2006-05-15T00:00:00Z"
    # text = "Carlo Azeglio Ciampi |S: +1999-05-18T00:00:00Z"  # 测试没有 |E 的情况

    # 正则表达式匹配
    pattern = r"^(.*?)\s*\|S:\s*([\+\d\-T:Z]+)(?:\s*\|E:\s*([\+\d\-T:Z]+))?"

    # 执行匹配
    match = re.match(pattern, implict_time_event)

    # 提取结果
    if match:
        name = match.group(1)  # 匹配名字
        start_date = match.group(2)  # 匹配 |S: 后的日期
        end_date = match.group(3) if match.group(3) else None  # 匹配 |E: 后的日期或设置为 None
    else:
        assert False, f"Failed to match the pattern in the text: {text}"
    
    return name, start_date.split('T')[0].lstrip('+'), end_date.split('T')[0].lstrip('+') if end_date is not None else None


def main():
    original_questions = load_json('/path/project/my_data/question/up2dated_qa.json')
    implict_time_event = load_json('/path/project/my_data/ssgg_newdata/implict_time_event.json')

    data_to_analyze = {}
    answer = {}
    for item in list(original_questions.keys()):
        answer[item] = prepare_answers(item, original_questions, EXCEPTIONS)

    unique_answers = set()
    un_unique_answers = set() # 去掉国家历史上重复出现的人名，避免在提问时产生歧义。

    for domain in original_questions:
        if domain not in data_to_analyze:
            data_to_analyze[domain] = {}
        for element in tqdm(original_questions[domain], desc=domain):
            if element not in data_to_analyze[domain]:
                data_to_analyze[domain][element] = {}
            if domain in ["countries_byGDP", "organizations"]:
                for attribute in original_questions[domain][element]:
                    if attribute not in data_to_analyze[domain][element]:
                        data_to_analyze[domain][element][attribute] = {}

                    questions = original_questions[domain][element][attribute]["questions"]
                    answers = original_questions[domain][element][attribute]["answers"]

                    for answer in answers:
                        answer_name, _, _ = prepare_time_event(answer)
                        if answer_name not in unique_answers:
                            unique_answers.add(answer_name)
                        else:
                            un_unique_answers.add(answer_name)

                    data_to_analyze[domain][element][attribute].update({"questions": questions})
                    name_to_remove = []
                    for name, events in implict_time_event[domain][element][attribute]['answers'].items():
                        for event in events[:]:
                            if event['main_type'] != "countries_byGDP":
                                events.remove(event)
                            elif event['entity'] == element:
                                events.remove(event)
                            else:
                                item_name, item_start_date, item_end_date = prepare_time_event(event['item'])
                                time_list = [item_start_date, item_end_date] if item_end_date is not None else [item_start_date]
                                if any(date[5:7] == "00" or date[8:10] == "00" for date in time_list) or item_name in un_unique_answers:
                                    events.remove(event)
                                    continue
                                if item_end_date is not None:
                                    format_item_start_date = datetime.strptime(item_start_date, '%Y-%m-%d')
                                    format_item_end_date = datetime.strptime(item_end_date, '%Y-%m-%d')
                                    if format_item_start_date >= format_item_end_date or (format_item_end_date - format_item_start_date).days < 365:
                                        events.remove(event)
                        if len(events) < 1:
                            name_to_remove.append(name)
                    
                    for name in name_to_remove:
                        implict_time_event[domain][element][attribute]['answers'].pop(name)

                    assert len(implict_time_event[domain][element][attribute]['answers']) >= 1, f"Failed to find a valid event for {domain} {element} {attribute}"

                    data_to_analyze[domain][element][attribute].update({"answers": implict_time_event[domain][element][attribute]['answers']})
            else:
                questions = original_questions[domain][element]["questions"]
                answers = original_questions[domain][element]["answers"]

                for answer in answers:
                    answer_name, _, _ = prepare_time_event(answer)
                    if answer_name not in unique_answers:
                        unique_answers.add(answer_name)
                    else:
                        un_unique_answers.add(answer_name)

                data_to_analyze[domain][element].update({"questions": questions})
                name_to_remove = []
                for name, events in implict_time_event[domain][element]['answers'].items():
                    for event in events[:]:
                        if event['main_type'] != "countries_byGDP":
                            events.remove(event)
                        elif event['instance'] == element:
                            events.remove(event)
                        else:
                            item_name, item_start_date, item_end_date = prepare_time_event(event['item'])
                            time_list = [item_start_date, item_end_date] if item_end_date is not None else [item_start_date]
                            if any(date[5:7] == "00" or date[8:10] == "00" for date in time_list) or item_name in un_unique_answers:
                                events.remove(event)
                                continue
                            if item_end_date is not None:
                                format_item_start_date = datetime.strptime(item_start_date, '%Y-%m-%d')
                                format_item_end_date = datetime.strptime(item_end_date, '%Y-%m-%d')
                                if format_item_start_date >= format_item_end_date or (format_item_end_date - format_item_start_date).days < 365:
                                    events.remove(event)
                    if len(events) < 1:
                        name_to_remove.append(name)
                        
                for name in name_to_remove:
                    implict_time_event[domain][element]['answers'].pop(name)
                assert len(implict_time_event[domain][element]['answers']) >= 1, f"Failed to find a valid event for {domain} {element}"

                data_to_analyze[domain][element].update({"answers": implict_time_event[domain][element]['answers']})
    
    dump_json('/path/project/my_data/ssgg_newdata/country_event.json', data_to_analyze)

if __name__ == "__main__":
    main()