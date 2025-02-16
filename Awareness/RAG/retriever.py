from functools import lru_cache
import json
import os
from pprint import pprint
import random
import sys
import time
import numpy as np
import requests
import urllib
sys.path.append(os.path.abspath("./"))


SEARCH_ACTION_DESC = {
    "book":         "The API provides access to medical knowledge resource including various educational resources and textbooks.",
    "guideline":    "The API provides access to clinical guidelines from leading health organizations.",
    "research":     "The API provides access to advanced biomedical research, facilitating access to specialized knowledge and resources.",
    "wiki":         "The API provides access to general knowledge across a wide range of topics.",
    "graph":        "The API provides a structured knowledge graph that connects medical definitions and related terms.",

    # "mixtext":      "The API provides access to 1. clinical guidelines from leading health organizations; 2. advanced biomedical research, facilitating access to specialized knowledge and resources; 3. medical knowledge resource including various educational resources and textbooks; 4. general knowledge across a wide range of topics.",
}
SEARCH_ACTION_PARAM = {
    "book":         r"{search_query0}",
    "guideline":    r"{search_query0}",
    "research":     r"{search_query0}",
    "wiki":         r"{search_query0}",
    "graph":        r"{medical_term0} , {query_for_term0} (Each query should use , to separate the {medical_term} and {query_for_term})",

    # "mixtext":      r"{search_query0}",
}

session = requests.Session()

class Retriever:
    def __init__(self, topk):
        self.topk = topk

    # @lru_cache(maxsize=10000)
    def run(self, units, add_query=False):
        args = []
        for u in units:
            assert u["source"] in SEARCH_ACTION_DESC
            args.append({"source": u["source"], "query": u["query"], "topk": self.topk})
        ##### Run Search #####
        try_number = 10
        for try_index in range(try_number):
            try:
                params = {
                    "secret": "cherry",
                    "args": json.dumps(args, ensure_ascii=False)
                }
                # print(json.dumps(params, indent=2))
                encoded_params = urllib.parse.urlencode(params)
                search_url = f"http://150.158.133.87:10000/?{encoded_params}"
                # print(search_url)
                t1 = time.time()
                search_result = session.get(search_url, timeout=300).content.decode('utf-8')
                # print('single:', time.time() - t1)
                search_result = json.loads(search_result)["success"]
                assert len(search_result) == len(args)
                break
                
            except Exception as e:
                if try_index == try_number - 1:
                    raise ValueError(f"Error in Search: {search_url} Error: {e}")
                print(f"Error in Search: {search_url} Error: {e}")
                time.sleep(6)
        ######################
        for index, ar in enumerate(args):
            if add_query:
                single_text = f"## source: {ar['source']}; query: {ar['query']}\n"
            else:
                single_text = f""
            if len(search_result[index]) > 0:
                single_text += "\n".join([doc['para'] for doc in search_result[index]])
            else:
                single_text += "There are no searching results."
            single_text = single_text.strip()
            ar["docs"] = single_text

        return args


if __name__ == "__main__":
    retriever = Retriever(topk=1)
    units = [
        {"source": "wiki", "query": "The current Chief Executive Officer of Intel is"},
    ]
    print(json.dumps(retriever.run(units), indent=2))
