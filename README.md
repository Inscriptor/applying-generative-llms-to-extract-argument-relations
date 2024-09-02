# applying-generative-llms-to-extract-argument-relations
Дополнительные материалы к статье **"Applying Generative Neural Networks to Extract Argument Relations 
from Scientific Communication Texts"**.

## Примеры инструкций к модели

Примеры инструкций находятся в директории ```data/scicorp/prompts```.
Для получения готового промпта для модели необходимо подцепить к инструкции входные данные, т.е.
пару предложений, между которыми требуется предсказать наличие или отсутствие аргументативной связи,
при необходимости --- их ближайший контекст, и обернуть в шаблон, ожидаемый моделью.

Для понимания шаблона см. файлы ```config/saiga_system_prompt.json``` и ```src/util/saiga/bot.py```.
Шаблон соответствует инструкции в карточке модели на [HuggingFace](https://huggingface.co/IlyaGusev/saiga_7b_lora).

## Исходники

Имеется два основных исполняемых файла, соответствующих двум применявшимся стратегиям:
```src/saiga/io-prompting.py``` и ```src/saiga/vote-prompting.py```. В текущем виде для запуска требуется
подключение к серверу **ClearML**, т.к. ожидается, что данные лежат там. Ожидаемый датасет должен включать
три файла: ```data.csv```, ```paragraphs.csv``` и ```all_sentences.csv```. Как они должны быть устроены
см. в функции ```src/util/common.py:prepare_dataset()```.