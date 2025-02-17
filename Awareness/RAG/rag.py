import json,ipdb
from retriever import Retriever
import argparse,sys
sys.path.append('/path/project/')
from tqdm import tqdm
from models_output.utils import EXCEPTIONS, load_json, dump_json

def main():
    retriever = Retriever(topk=1)

    parser = argparse.ArgumentParser(
        description="Generate context using GPT-4",
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
        "--time-travel_file",
        metavar="FILE_PATH",
        type=str,
        default="/path/project/models_output/temporal_awareness/Awareness/time_travel_data.json",
        help="Path to the file containing Q&A.",
    )
    parser.add_argument(
        "--passages-path",
        metavar="FILE_PATH",
        type=str,
        default="/path/project/my_data/passage/passages.json",
        help="Path to the file containing the passages collected from Wikipedia.",
    )

    args = parser.parse_args()
    original = load_json(args.grc_path)
    original_passage = load_json(args.passages_path)
    time_info = load_json(args.time_travel_file)

    category = list(original.keys())
    item = dict()

    for category, elements in tqdm(original.items(),desc="Categories"):
        item[category] = {}
        for element, attributes in tqdm(elements.items(), desc=f"Generating Answ. for {category}"):
            item[category][element] = {}
            if category in ["countries_byGDP", "organizations"]:
                for attribute, grc_elem in attributes.items():
                    item[category][element][attribute] = {}

                    past_date = time_info[category][element][attribute]["time_travel_date"]
                    query = f"Who is the {attribute} in {past_date}?" if category == 'countries_byGDP' else f"Who is the {attribute} of the {element} on {past_date}?"

                    units = [
                        {"source": "wiki", "query": query},
                    ]
                    result = retriever.run(units)
                    time_info[category][element][attribute]["future_news"]['text'] = result[0]['docs']
                    time_info[category][element][attribute]["future_news"].pop('name')
                    time_info[category][element][attribute]["future_news"].pop('dates')
                    original_passage[category][element][attribute]["matches"]['rag_results'] = {'text': result[0]['docs']}
            else:
                    past_date = time_info[category][element]["time_travel_date"]
                    query = f"Who is the Chief Executive Officer of {element} on {past_date}?" if category == 'companies_byRevenue' else f"Which team did  {element} serve for on {past_date}?"

                    units = [
                        {"source": "wiki", "query": query},
                    ]
                    result = retriever.run(units)
                    time_info[category][element]["future_news"]['text'] = result[0]['docs']
                    time_info[category][element]["future_news"].pop('name')
                    time_info[category][element]["future_news"].pop('dates')
                    original_passage[category][element]["matches"]['rag_results'] = {'text': result[0]['docs']}

    dump_json('/path/project/models_output/temporal_awareness/Awareness/RAG/rag_time_travel.json', time_info)
    dump_json('/path/project/models_output/temporal_awareness/Awareness/RAG/rag_passagess.json', original_passage)


if __name__ == "__main__":
    main()