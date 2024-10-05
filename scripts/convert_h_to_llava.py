import json
import os
import fire
import re
from convert_sqa_to_llava_base_prompt import build_prompt_chatbot


def convert_to_llava(base_dir):
    file = open(os.path.join(base_dir, "hellaswag.jsonl"))
    problems = []
    for line in file.readlines():
        dic = json.loads(line)
        problems.append(dic)

    target_format = []
    for prob_id, problem in enumerate(problems):
        context = problem['ctx']
        options = problem['ending_options']
        options = '\n'.join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
        
        target_format.append({
            "id": prob_id,
            "conversations": [
                {'from': 'human', 'value': f"{context}\n{options}"},
                {'from': 'gpt', 'value': "hellaswag dataset evaluation"},
            ],
        })

    print(f'Number of samples: {len(target_format)}')

    with open(os.path.join(base_dir, f"llava_hellaswag.json"), "w") as f:
        json.dump(target_format, f, indent=2)




if __name__ == "__main__":
    convert_to_llava("./scripts")
