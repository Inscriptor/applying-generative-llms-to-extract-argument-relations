import copy
from util.saiga.bot import Conversation


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def generate(
    model,
    tokenizer,
    prompts,
    generation_config,
    debug: bool = True
):
    data = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        sample_output = sample_output.replace("</s>", "").strip()
        if debug:
            print(tokenizer.decode(sample_input_ids, skip_special_tokens=True))
            print(sample_output)
            print()
        outputs.append(sample_output)
    return outputs


def predict_saiga_zero_shot(
    model,
    tokenizer,
    generation_config,
    template_path,
    prompts,
    max_prompt_tokens: int = None,
    debug: bool = False
):
    default_conversation = Conversation.from_template(template_path)
    clean_prompts = []
    for prompt in prompts:
        conversation = copy.deepcopy(default_conversation)
        conversation.add_user_message(prompt)
        prompt = conversation.get_prompt(tokenizer, max_tokens=max_prompt_tokens)
        clean_prompts.append(prompt)
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompts=clean_prompts,
        generation_config=generation_config,
        debug=debug
    )


def respond_to(model, tokenizer, generation_config, template_path, prompt):
    """
    Same as predict_saiga_zero_shot, but takes only a single prompt.
    """
    return predict_saiga_zero_shot(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        template_path=template_path,
        prompts=[prompt]
    )[0]
