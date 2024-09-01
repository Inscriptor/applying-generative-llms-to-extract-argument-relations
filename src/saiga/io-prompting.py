import os
import sys
sys.path.insert(0, '..')

import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datetime import datetime

from clearml import Task, TaskTypes, Dataset

from util.saiga.load import load_saiga
from util.saiga.misc import gen_batch, predict_saiga_zero_shot
from util.common import prepare_dataset, quality_metrics, clean_json_binary_predictions


def eval(records, responses):
    """
    Take lists of records and LLM responses and evaluate prediction quality.
    This function computes the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - All above metrics computer on a data subset with positive / negative samples
      ratio is equal to 0.15.
    - Number of invalid responses, i.e. such that did not contain any prediction that
      we would be able to correctly extract.
    
    @param records:   list of records, where each record is a pair of sentences and label
    @param responses: list of raw responses got from a LLM
    """
    labels, predictions = [], []
    tpc = 0
    fpc = 0
    for record, response in zip(records, responses):
        prediction = clean_json_binary_predictions(response, logging.getLogger('parse-output'))
        label = record["label"]
        labels.append(label)
        predictions.append(prediction)
        if prediction == 1:
            if label == 1:
                tpc += 1
            else:
                fpc += 1

    acc, prec, rec, f1 = quality_metrics(labels=labels, predictions=predictions)
    acc_, prec_, rec_, f1_ = quality_metrics(labels=labels, predictions=predictions, ratio=.15)
    irc = len([p for p in predictions if np.isnan(p)])
    return acc, prec, rec, f1, acc_, prec_, rec_, f1_, irc, tpc, fpc


def log_plots(groups, accuracy, prec, recall, f1, title):
    _group_num = len(groups)
    bar_width = 0.22  # the width of the bars
    x_pos = np.arange(_group_num)  # the label locations
    multiplier = 0
    #if _group_num <= 3:
    #    fig_width = 6.5  # the width of the figure
    #else:
    #    fig_width = 6.5 + (_group_num - 3) * 2.2

    fig, ax = plt.subplots(layout="constrained")
    #fig.set_figheight(5.6)
    #fig.set_figwidth(fig_width)

    _data_object = {
        'Accuracy': accuracy,
        'Precision': prec,
        'Recall': recall,
        'F1-score': f1
    }

    for group, values in _data_object.items():
        offset = bar_width * multiplier
        rects = ax.bar(x_pos + offset, values, bar_width, label=group)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Metric value')
    ax.set_title(title)
    ax.set_xticks(x_pos + bar_width, groups)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 1)

    plt.show()


def main(conf_file, data_name, prompt_file, model_name_, test_mode, load_checkpoint):
    """
    Iterate given prompt templates. For each template make prompt list using data records.
    Record is a pair of sentences and their contexts. Depending on the run settings, contexts
    may or may not be included to a prompt.
    This program implements simple I/O prompting when a LLM model is asked if two sentences
    are related to each other in a specific way. Model answers are stored, cleaned and interpreted,
    and then compared to labels.
    Results are the classification metrics: accuracy, precision, recall and F1. Each metric
    computed twice: on a full dataset and on its subset where positive samples rates to
    negative ones in a specified ratio (usually 0.15).
    
    @param conf_file      : path to the configuration file
    @param data_name      : name of the dataset; ClearML Dataset is defined by its name and
                            project; here we assume that project name is the same as in the Task,
                            so we need name of the dataset only; if dataset name if not specified
                            then use dataset ID specified in the configuration file; dataset is
                            expected to contain at least 3 CSV files:
                            data.csv, all_sentences.csv and parahraphs.csv
    @param prompt_file    : path to the file with prompt templates; if not defined use config.prompts
    @param model_name_    : name of the LLM model; if not defined use config.model_settings.model_name
    @param test_mode      : if true use a dataset of a size of 20 records
    @param load_checkpoint: path to checkpoint (not used now)
    """
    Task.set_random_seed(random_seed=123456789)
    project_name='ArgMining'

    task = Task.init(
        project_name=project_name,
        task_name="I/O prompting",
        task_type=TaskTypes.inference,
        reuse_last_task_id=False)
    task.add_tags(["i/o prompting", "mistral", "peft", "saiga"])

    if not os.path.isfile(conf_file):
        print(f"Cannot open {conf_file}: either it does not exist or unaccessible.")
        task.mark_failed()
        return

    with open(conf_file) as cf:
        config = json.loads(cf.read())

    logging.basicConfig(
        filename=os.path.join(config["logdir"], 'io-prompting.log'),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.DEBUG
    )

    task.set_parameters_as_dict(config)

    if data_name:
        read_data_from = Dataset.get(
            dataset_project=project_name,
            dataset_name=data_name,
            alias="SCICORP-2").get_local_copy()
    else:
        read_data_from = Dataset.get(dataset_id=config["datasetCML"], alias="SCICORP-2").get_local_copy()

    print('Prepare data', end='')

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

    with open(read_prompts_from, 'r') as tf:
        prompt_templates = json.loads(tf.read())

    now_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = os.path.join(config["outdir"], f'{now_time}-io-prompting.csv')

    use_context = config["use_context"]

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
    
    accs_ = []
    accs_15 = []
    precs = []
    precs_15 = []
    recs = []
    recs_15 = []
    f1s = []
    f1s_15 = []
    invalid_response_counts = []
    tp_counts = []
    fp_counts = []
    pt_names = []

    for name, template in prompt_templates.items():
        prompt_template = template['prompt']
        if use_context:
            prompts = [f'{prompt_template}\n\n#### Входные данные:\ncontext:{r["context"]}\npremise:{r["premise"]}\nconclusion:{r["conclusion"]}' for r in records]
        else:
            prompts = [f'{prompt_template}\n\n#### Входные данные:\npremise:{r["premise"]}\nconclusion:{r["conclusion"]}' for r in records]
        responses = []
        for batch in tqdm(gen_batch(prompts, batch_size), total=len(prompts) // batch_size + 1):
            responses.extend(predict_(batch))
        
        pt_names.append(name)

        ac, pr, rc, f1, ac_, pr_, rc_, f1_, irc, tpc, fpc = eval(records, responses)
        accs_.append(ac)
        accs_15.append(ac_)
        precs.append(pr)
        precs_15.append(pr_)
        recs.append(rc)
        recs_15.append(rc_)
        f1s.append(f1)
        f1s_15.append(f1_)
        invalid_response_counts.append(irc)
        tp_counts.append(tpc)
        fp_counts.append(fpc)

    report_metric = max(f1s)
    task.get_logger().report_single_value(name='MAX F1', value=report_metric)
    log_plots(
        groups=pt_names,
        accuracy=accs_,
        prec=precs,
        recall=recs,
        f1=f1s,
        title="Classification metrics on a full dataset"
    )

    pd.DataFrame(data={
        'prompt': pt_names,
        'acc': accs_,
        'prec': precs,
        'rec': recs,
        'f1': f1s,
        'acc15': accs_15,
        'prec15': precs_15,
        'rec15': recs_15,
        'f1-15': f1s_15,
        'bad_response': invalid_response_counts,
        'TP': tp_counts,
        'FP': fp_counts
    }).to_csv(output, encoding='utf8', index=False)

    task.close()


if __name__ == '__main__':
    # Example:
    # python3 io-prompting.py 
    #                         --cfg "<config.json>" 
    #                         --ds "<dataset-name>" 
    #                         --ps "<path-to-prompts>" 
    #                         --model "<model-name>" 
    #                        [--text-mode]
    #                        [--checkpoint "<checkpoint-file>"]

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, type=str, dest='config', help='configuration file (JSON)')
    parser.add_argument('--ds', type=str, dest='ds', help='name of the dataset (from ClearML storage)')
    parser.add_argument('--ps', type=str, dest='prompts', help='path to the JSON file with prompts')
    parser.add_argument('--model', type=str, dest='model', help='LLM model name (if not specified get from config)')
    parser.add_argument('--checkpoint', type=str, default=None, dest='checkpoint', help='load checkpoint from a file (not used yet)')
    parser.add_argument('--test-mode', action='store_true', dest="test", help='if set run script on a small part of a dataset (20 records only)')

    args = parser.parse_args()

    main(conf_file=args.config,
         prompt_file=args.prompts,
         data_name=args.ds,
         model_name_=args.model,
         test_mode=args.test,
         load_checkpoint=args.checkpoint)
