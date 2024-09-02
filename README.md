# applying-generative-llms-to-extract-argument-relations
Additional materials to the paper **"Applying Generative Neural Networks to Extract Argument Relations 
from Scientific Communication Texts"**.

## Model instruction samples

The instruction examples are located in the directory `data/scicorp/prompts`.
To generate a complete prompt for the model, you need to attach the input data to the instruction.
The expected input is the pair of sentences for which you need to predict the presence or absence of
an argumentative relationship, and, if necessary, their immediate context.
Then, wrap the resulting prompt in the template format expected by the model.

To view the template, refer to the files `config/saiga_system_prompt.json` and `src/util/saiga/bot.py`.
The template follows the instruction format specified in the model's card on [HuggingFace](https://huggingface.co/IlyaGusev/saiga_7b_lora).
These files will guide you on how to structure the input data and instructions according to the model's
expected format, ensuring that the prompts are correctly processed.

## Source code

There are two main executable files corresponding to the two strategies used:

1. `src/saiga/io-prompting.py`
2. `src/saiga/vote-prompting.py`

Currently, to run these scripts, a connection to the **ClearML** server is required, as the data is expected
to be hosted there. The expected dataset should include the following three files:
`data.csv`, `paragraphs.csv`, and `all_sentences.csv`.

To understand the structure these files should have, refer to the function `prepare_dataset()` located
in `src/util/common.py`. This function provides the guidelines for how the dataset should be organized
for proper processing by the scripts.