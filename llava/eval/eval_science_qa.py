import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            pred = {'text': 'FAILED', 'prompt': 'Unknown'}
            pred_text = 'FAILED'
        else:
            pred = predictions[prob_id]
            pred_text = pred['text']

        if pred_text in args.options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        else:
            pattern = re.compile(r'The answer is ([A-Z]).')
            res = pattern.findall(pred_text)
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
            else:
                answer = "FAILED"

        pred_idx = get_pred_idx(answer, prob['choices'], args.options)
        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': args.options[prob['answer']],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
            "grade":prob["grade"],
            "subject": prob["subject"],
            "no_context": True if (not prob['hint'] and not prob['image']) else False,
            "has_text": True if prob['hint'] else False,
            "has_image": '<image>' in pred['prompt'],
            "has_text_image": True if (prob['hint'] and prob['image']) else False,
        }

        sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        if pred_idx == prob['answer']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    ###### IMG ######
    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######
    #################


    scores = {
        'acc_natural':
        len([x for x in results['correct'] if "natural science" in x['subject']]) / (len([x for x in results['correct'] if "natural science" in x['subject']])+len([x for x in results['incorrect'] if "natural science" in x['subject']])),
        'acc_social':
        len([x for x in results['correct'] if "social science" in x['subject']]) / (len([x for x in results['correct'] if "social science" in x['subject']])+len([x for x in results['incorrect'] if "social science" in x['subject']])),
        'acc_language':
        len([x for x in results['correct'] if "language science" in x['subject']]) / (len([x for x in results['correct'] if "language science" in x['subject']])+len([x for x in results['incorrect'] if "language science" in x['subject']])),
        'acc_has_text':
        len([x for x in results['correct'] if x['has_text']])/ (len([x for x in results['correct'] if x['has_text']])+len([x for x in results['incorrect'] if x['has_text']])),
        'acc_has_image':
        len([x for x in results['correct'] if x['has_image']])/ (len([x for x in results['correct'] if x['has_image']])+len([x for x in results['incorrect'] if x['has_image']])),
        'acc_no_context':
        len([x for x in results['correct'] if x['no_context']])/ (len([x for x in results['correct'] if x['no_context']])+len([x for x in results['incorrect'] if x['no_context']])),
        'acc_grade_1_6':
        len([x for x in results['correct'] if x['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']])/ (len([x for x in results['correct'] if x['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']])+len([x for x in results['incorrect'] if x['grade'] in ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']])),
        'acc_grade_7_12':
        len([x for x in results['correct'] if x['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']])/ (len([x for x in results['correct'] if x['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']])+len([x for x in results['incorrect'] if x['grade'] in ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']]))
    }
    print(scores)
    #################
    print(f'\nTotal: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')
    
    sqa_results['acc'] = correct / total * 100  
    sqa_results['correct'] = correct
    sqa_results['count'] = total
    sqa_results['scores'] = scores 
    sqa_results['img_acc'] = multimodal_correct / multimodal_total * 100

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
