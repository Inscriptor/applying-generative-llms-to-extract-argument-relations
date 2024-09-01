import re
import math
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from json_repair import repair_json
from sklearn.metrics import accuracy_score

def quality_metrics(labels, predictions, ratio=1.):
    """
    Compute classification quality metrics, such as:
    - accuracy
    - precision
    - recall
    - f1-score
    It is assumed that label could be 1 or 0.
    :param ratio       : 
    :param labels      :
    :param predictions :
    """
    df = pd.DataFrame(data={'label': labels, 'prediction': predictions})
    df = df[df.prediction >= 0]
    if ratio < 1.:
        # take positive samples in number of floor(<num_negatives * ratio)
        minor_size = math.floor((df.shape[0] - df.label.sum()) * ratio)
        df = pd.concat([
            df[df.label == 0], df[df.label == 1].sample(n=minor_size, random_state=123456789)
        ])
    df_matched = df[df.label == df.prediction]
    accuracy = df_matched.shape[0] / df.shape[0]
    tp = df_matched[df_matched.prediction == 1].shape[0]
    precision = tp / df[df.prediction == 1].shape[0]
    recall = tp / df[df.label == 1].shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    
    return accuracy, precision, recall, f1


def clean_and_eval(records, responses, clean_func, ratio=.15, saveto=None):
    """
    Clean model responses, then compute and print all metrics:
    accuracy, presicion, recall and F1-score. Designed for binary classification only.
    :param records    : List of records from a dataset. Each record is expected to be an object (dict)
    :param responses  : List of model responses
    :param clean_func : Response cleaning function
    :param ratio      : If not equal to 1, take positive samples in a certain ratio to negative ones when computing metrics
    :param saveto     : If not None then save metrics to a specified CSV file
    """
    labels, predictions = [], []
    for record, response in zip(records, responses):
        prediction = clean_func(response)
        # record["prediction"] = prediction
        label = record["label"]
        if label != -1:
            labels.append(label)
            predictions.append(prediction)
    
    print(f'ACC-SCORE\nAccuracy  = {accuracy_score(labels, predictions)}\n=============================\n\n')
    acc, prec, rec, f1 = quality_metrics(labels, predictions)
    print(f'Q-SCORE-1\nAccuracy  = {acc}\nPrecision = {prec}\nRecall    = {rec}\nF1        = {f1}\n=============================\n\n')
    acc, prec, rec, f1 = quality_metrics(labels, predictions, ratio=ratio)
    print(f'Q-SCORE-{ratio}\nAccuracy  = {acc}\nPrecision = {prec}\nRecall    = {rec}\nF1        = {f1}')

    if saveto:
        pd.DataFrame(data={'label': labels, 'prediction': predictions}
                     ).to_csv(saveto, encoding='utf8', index=False)



def sanitize_unescaped_quotes_and_load_json_str(s: str, strict=False) -> dict:  # type: ignore
    """
    Processing a string looking for unescaped quotes and then attempt to parse it as JSON.
    """
    js_str = s
    prev_pos = -1
    curr_pos = 0
    while curr_pos > prev_pos:
        # after while check, move marker before we overwrite it
        prev_pos = curr_pos
        try:
            return json.loads(js_str, strict=strict)
        except json.JSONDecodeError as err:
            curr_pos = err.pos
            if curr_pos <= prev_pos:
                # previous change didn't make progress, so try to repair by other means
                # and if it does not help then deal with an error somewhere outside
                return json.loads(repair_json(js_str), strict=strict)

            # find the previous " before e.pos
            prev_quote_index = js_str.rfind('"', 0, curr_pos)
            # escape it to \"
            js_str = js_str[:prev_quote_index] + "\\" + js_str[prev_quote_index:]


def pretty_output(*unnamed, **named):
    """
    Get an unknown list of values and make a pretty output of them.
    @param *unnamed: list of arguments that are to be output in a row;
    @param **named : kwargs that are to be output in a column as: <Name>: <Value>
    """
    if (len(unnamed) > 0):
        print('Values: ', end='')
        for val in unnamed:
            print(f'{val} ', end='')
        print('')
    if (len(named.keys()) > 0):
        max_name_len = max([len(key) for key in named.keys()])
        for key, value in named.items():
            key_str = f'{key}:'.ljust(max_name_len)
            print(f'{key_str} {value}')


def clean_json_binary_predictions(llm_response_json_string, logger=None):
    """
    Extract binary prediction from a JSON-formatted string
    returned by a LLM in response to a request prompt. Since this is a
    binary prediction we want to extract, the function is trying to
    parse input string as JSON, reparing it if necessary, and extract
    prediction, made by a model, as 0 or 1.
    If function is not able to extract any sensible information from
    the input string, it returns NaN.
    If logger is not None, log information and warnings

    @param llm_response_json_string : string returned by a LLM
    @param logger                   : logger (None by default)
    @return                         : Return 0 or 1 depending on
                                      prediction a model made, or NaN
                                      if no prediction was extracted.
    """
    ob = re.escape('{')  # open curly brace
    cb = re.escape('}')  # close curly brace
    if llm_response_json_string == '':
        # if model returned en empty string, it is interpreted as a negative prediction
        return 0
    # remove newline symbols and search for a substring enclosed in curly braces
    fn = re.search(ob + '(.*)' + cb, str(llm_response_json_string).replace('\n', ''))
    # if fn is not empty, try to parse something found inside curly braces as JSON
    if fn:
        try:
            return sanitize_unescaped_quotes_and_load_json_str(s=fn.group(0))['prediction']
        except KeyError:
            if logger is not None:
                logger.warning(f'Decoded JSON with no goal field: {str(llm_response_json_string)}')
            return np.nan;
    # otherwise try to extrace prediction directly
    fn = re.search(r'(0|1)',str(llm_response_json_string).replace('\n', ''))
    if fn:
        return int(fn.group(0))
    elif logger is not None:
        logger.info(f'Cannot parse a response: {str(llm_response_json_string)}')
    return np.nan


def prepare_dataset(labels_csv, sentences_csv, paragraphs_csv, truncate=False):
    """
    Making a dataset consisting of the following columns:
     - premise:     premise sentence
     - conclusion:  conclusion sentence
     - context:     immediate context of a pair composed of paragraphs containing both sentences of a pair
     - label:       label of a pair; it is equal to 1 if there is an argumentation relashionship between 
                    premise and conclusion and 0 otherwise

    from three CSV files of specified structure.
                    
    @param labels_csv     : path to the CSV file containing data records;
                            among others it is expected to include fields 
                            'premise', 'conclusion', 'label', 'text',
                            'premise_sentence_id' and 'conclusion_sentence_id'
    @param sentences_csv  : path to the CSV file with sentece-to-paragraph correspondence
    @param paragraphs_csv : path to the CSV file where paragraphs contents are stored
    @return               : return resulting dataset as a list of records
    """
    labels = pd.read_csv(labels_csv)
    sentns = pd.read_csv(sentences_csv).set_index(['text', 'sentence_id'])
    parags = pd.read_csv(paragraphs_csv, usecols=['text', 'content', 'id', 'span']).set_index(['text', 'id'])
   
    contexts = []

    if truncate == False:
        mainframe = labels
        total_cnt = labels.shape[0]
    else:
        mainframe = labels.sample(n=truncate, random_state=123456789)
        total_cnt = truncate
    
    for _, row in tqdm(mainframe.iterrows(), total=total_cnt):
        txtid = row['text']
        prsid = row['premise_sentence_id']
        ccsid = row['conclusion_sentence_id']
        prpid = sentns.loc[txtid, prsid]['paragraph_id']
        ccpid = sentns.loc[txtid, ccsid]['paragraph_id']
        if prpid == ccpid:
            context = parags.loc[txtid,prpid]['content']
        else:
            context = f'{parags.loc[txtid,prpid]["content"]}\n{parags.loc[txtid,ccpid]["content"]}'
        contexts.append(context)

    data = mainframe[['premise', 'conclusion', 'label']].copy()
    data['context'] = contexts

    return data[['premise','conclusion','context','label']].to_dict("records")
