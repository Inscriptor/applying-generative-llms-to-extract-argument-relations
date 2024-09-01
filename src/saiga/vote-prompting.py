import os
import sys
sys.path.insert(0, '..')

import re
import json
import logging
import argparse
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score

from clearml import Task, TaskTypes, Dataset

from util.saiga.load import load_saiga
from util.saiga.misc import gen_batch, predict_saiga_zero_shot
from util.common import clean_json_binary_predictions, prepare_dataset, quality_metrics


def make_prompt(prompt_template, record):
    return f'{prompt_template}\n\n#### Входные данные:\ncontext:{record["context"]}\npremise:{record["premise"]}\nconclusion:{record["conclusion"]}'


def clean_json_resposes(response):
    return clean_json_binary_predictions(response, logging.getLogger('clean_json_resposes'))


def make_rcb_prompt(prompt_template, record):
    SCICORP_TAIL_CHARS = re.escape('.;,:!? ')
    return prompt_template.format(premise=record['premise'],
                                  question=re.sub(f'[{SCICORP_TAIL_CHARS}]+$', '', record['conclusion']) + '?')


def clean_rcb_responses(response):
    SCICORP_YES_RE = re.compile(
        r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(да|верно|вероятно)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    SCICORP_NO_RE = re.compile(
        r"^[^\w]*(Выходные данные|Выход|Ответ|Оценка)?[^\w]*(нет|неверно|неверное|невероятно|не)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    is_contradiction = bool(SCICORP_YES_RE.match(response))
    is_entailment = bool(SCICORP_NO_RE.match(response))
    if is_contradiction:
        return 1
    if is_entailment:
        return 0
    return 0


def process_separate(records, prompt_template, predict_func, n, batch_size, make_prompt_func, clean_response_func, checkpoint):
    # Make prompts
    prompts = [make_prompt_func(prompt_template, record) for record in records]
    # List of lists [preds_1, preds_2, ..., preds_n] where each preds_i is a list of length = r
    # This all_preds is a (n x r) tensor
    all_preds = checkpoint
    for i in range(n):
        responses = []
        for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
            responses.extend(predict_func(batch))
        predictions = [clean_response_func(response) for response in responses]
        all_preds.append(predictions)
        with open(f'../../checkpoints/vote-prompting-separate-{i}.json', 'w') as chk_file:
            json.dump(all_preds, chk_file)

    # Compute votes
    threshold = np.ceil(n / 2.)
    votes = [0 if np.nansum(preds) <= threshold else 1 for preds in np.transpose(all_preds)]
    # Compute probabilities
    probs = np.nanmean(all_preds, axis=0).tolist()

    return votes, probs


def process_mixed(records, prompt_templates, predict_func, n, batch_size):
    # How many predictions we generate for each prompt template
    k = np.ceil(n / len(prompt_templates)).astype(int)
    all_prompt_preds = []  # (m x k x r) tensor where m is number of prompts, r is length of the dataset
    for prompt_name in list(prompt_templates.keys()):
        print(f'Processing: {prompt_name}')
        prompt_info = prompt_templates[prompt_name]
        clean_ = globals()[prompt_info['clean']]
        make_ = globals()[prompt_info['make']]
        prompt_template = prompt_info['prompt']
        prompts = [make_(prompt_template, record) for record in records]
        all_preds = []  # (k x r)

        for _ in range(k):
            responses = []
            for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
                responses.extend(predict_func(batch))
            predictions = [clean_(response) for response in responses]
            all_preds.append(predictions)

        all_prompt_preds.append(all_preds)
        with open(f'../../checkpoints/vote-prompting-mixed-{prompt_name}.json', 'w') as chk_file:
            json.dump(all_prompt_preds, chk_file)

    # Merge (m x k x r) tensor to make (km x r)
    all_preds = np.concatenate(all_prompt_preds, axis=0)
    # Compute votes
    threshold = np.ceil(n / 2.)
    votes = [0 if np.nansum(preds) <= threshold else 1 for preds in np.transpose(all_preds)]
    # Compute probabilities
    probs = np.nanmean(all_preds, axis=0).tolist()

    return votes, probs


def main(conf_file, data_name, prompt_file, repeat_prompt, mixed_strat, model_name_, test_mode, load_checkpoint):
    """
    Given a list of prompts this function either runs a specified prompt template N times for each
    record or runs each prompt template approximately N / <number_of_templates> times. Thus, for each
    dataset record we get N predictions, which then used to obtain a pair of alternative resulting
    predictions. First is a voted prediction. If there is a list of 0s and 1s, we decide what result is
    by computing number of 0s and 1s. Whichever numbers are greater - this the final result is.
    The second is a probability prediction. It is equal to the mean value of the prediction list.
    Both voted and probability predictions are saved to output file.

    @param conf_file      : path to the configuration file
    @param data_name      : name of the dataset; if not defined use config.datasetCML
    @param prompt_file    : path to the file with prompt templates; if not defined use config.prompts
    @param repeat_prompt  : prompt template we should repeat
    @param mixed_strat    : if True ignore <repeat_prompt> and repeat each template ~ (N / <prompt_num>) times
    @param model_name_    : name of the LLM model; if not defined use config.model_settings.model_name
    @param test_mode      : if true use a dataset of a size of 20 records
    @param load_checkpoint: path to checkpoint
    """
    Task.set_random_seed(random_seed=123456789)
    project_name='ArgMining'

    task = Task.init(
        project_name=project_name,
        task_name="Vote prompting",
        task_type=TaskTypes.inference,
        reuse_last_task_id=False)
    task.add_tags(["mistral", "peft", "saiga", "vote"])

    if not os.path.isfile(conf_file):
        print(f"Cannot open {conf_file}: either it does not exist or unaccessible.")
        task.mark_failed()
        return
    
    # check if run mode is set correctly: either defined a prompt to repeat
    # or mixed strategy is chosen
    if repeat_prompt is None and not mixed_strat:
        print(f'Required settings missing: either set a prompt to repeat (--p) or set a mixed mode (--mixed)')
        task.mark_failed()
        return
    
    with open(conf_file) as cf:
        config = json.loads(cf.read())

    logging.basicConfig(
        filename=os.path.join(config["logdir"], 'vote-prompting.log'),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.DEBUG
    )

    task.set_parameters_as_dict(config)

    logger = logging.getLogger('main')

    if data_name:
        read_data_from = Dataset.get(
            dataset_project=project_name,
            dataset_name=data_name,
            alias="SCICORP-2").get_local_copy()
    else:
        read_data_from = Dataset.get(dataset_id=config["datasetCML"], alias="SCICORP-2").get_local_copy()

    if test_mode:
        records = prepare_dataset(
            labels_csv=os.path.join(read_data_from, 'data.csv'),
            sentences_csv=os.path.join(read_data_from, 'all_sentences.csv'),
            paragraphs_csv=os.path.join(read_data_from, 'paragraphs.csv'),
            truncate=20
        )
    else:
        records = prepare_dataset(
            labels_csv=os.path.join(read_data_from, 'data.csv'),
            sentences_csv=os.path.join(read_data_from, 'all_sentences.csv'),
            paragraphs_csv=os.path.join(read_data_from, 'paragraphs.csv')
        )

    if prompt_file:
        read_prompts_from = prompt_file
    else:
        read_prompts_from = config["prompts"]

    # open prompt file
    with open(read_prompts_from, 'r') as tf:
        prompt_templates = json.loads(tf.read())
        if not mixed_strat:
            try:
                prompt_template = prompt_templates[repeat_prompt]
                clean_func = globals()[prompt_template['clean']]
                make_func = globals()[prompt_template['make']]
            except KeyError:
                errmsg = f'Cannot find prompt template "{repeat_prompt}" in the prompt file.'
                logger.error(errmsg)
                print(errmsg)
                task.mark_failed()
                return


    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = os.path.join(config["outdir"], f'{now_time}-vote-prompting.json')

    repeats = config["repeat"]

    if load_checkpoint:
        with open(load_checkpoint, 'r') as checkpoint_file:
            checkpoint = json.loads(checkpoint_file.read())
    else:
        checkpoint = []

    if model_name_:
        model_name = model_name_
    else:
        model_name = config["model_settings"]["model_name"]

    template_path = "../../config/saiga_system_prompt.json"

    quantization = config["model_settings"]["quantization"]

    if quantization:
        model, tokenizer, generation_config = load_saiga(model_name)  # Perform quantization on a model
    else:
        model, tokenizer, generation_config = load_saiga(
            model_name, load_in_8bit=False, torch_dtype='float16'
        )

    # Common general settings
    generation_config.no_repeat_ngram_size = 64
    generation_config.temperature = 0.02

    max_new_tokens = config["model_settings"]["max_new_tokens"]
    batch_size = config["model_settings"]["batch_size"]

    def predict_(batch):
        generation_config.max_new_tokens = max_new_tokens
        return predict_saiga_zero_shot(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            template_path=template_path,
            prompts=batch,
            debug=False
        )
    

    if mixed_strat:
        votes, probabilities = process_mixed(
            records=records,
            prompt_templates=prompt_templates,
            predict_func=predict_,
            n=repeats,
            batch_size=batch_size
        )
    else:
        votes, probabilities = process_separate(
            records=records,
            prompt_template=prompt_template['prompt'],
            predict_func=predict_,
            n=repeats,
            batch_size=batch_size,
            make_prompt_func=make_func,
            clean_response_func=clean_func,
            checkpoint=checkpoint
        )

    all_labels = [record["label"] for record in records]
    _, _, _, f1 = quality_metrics(labels=all_labels, predictions=votes)
    roc_auc = roc_auc_score(all_labels, probabilities)
    task.get_logger().report_single_value(name="MAX F1", value=f1)
    task.get_logger().report_single_value(name="ROC AUC", value=roc_auc)

    # Save results
    try:
        with open(output, 'w') as out:
            json.dump({'votes': votes, 'probabilities': probabilities}, out)
    except:
        default_output = '../../data/scicorp/results/vote/latest-vote-prompting.json'
        logger.warning(f'Failed to open file {output}. Saved results to a defalt file {default_output}')
        with open(default_output, 'w') as out:
            json.dump({'votes': votes, 'probabilities': probabilities}, out)


if __name__ == '__main__':
    # Example:
    # python3 vote-prompting.py
    #                           --cfg "../../config/saiga_vote_prompting.json"
    #                           --ds "SciCorp2"
    #                           --ps "../../data/scicorp/prompts/mixed-prompts.json"
    #                           [--p "D2AC"]
    #                           [--model "models/saiga2_7b_no_context-2"]
    #                           [--mixed]
    #                           [--checkpoint "<path-to-checkpoint file>"]
    #                           [--test-mode]

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, type=str, dest='config', help='configuration file (JSON)')
    parser.add_argument('--ds', type=str, dest='ds', help='name of the dataset (from ClearML storage)')
    parser.add_argument('--ps', type=str, dest='prompts', help='path to the JSON file with prompts')
    parser.add_argument('--p', type=str, default=None, dest='prompt', help='prompt to repeat (default None)')
    parser.add_argument('--model', type=str, dest='model', help='LLM model name (if not specified get from config)')
    parser.add_argument('--mixed', action='store_true',
        help='if true ignore --p argument and run each prompt ~ (N / <total prompt number>) times on every example')
    parser.add_argument('--checkpoint', type=str, default=None, dest='checkpoint', help='load checkpoint from a file')
    parser.add_argument('--test-mode', action='store_true', dest="test", help='if set run script on a small part of a dataset (20 records only)')

    args = parser.parse_args()

    main(conf_file=args.config,
         data_name=args.ds,
         prompt_file=args.prompts,
         repeat_prompt=args.prompt,
         mixed_strat=args.mixed,
         model_name_=args.model,
         test_mode=args.test,
         load_checkpoint=args.checkpoint)
