# Machine-generated text detection

Repository for [SemEval2024](https://semeval.github.io/SemEval2024/) [task 8](https://github.com/mbzuai-nlp/SemEval2024-task8#data_format).

Team Unibuc - NLP at SemEval-2024 Task 8

Authors: Teodor-George Marchitan, Claudiu Creanga, Liviu Petrisor Dinu

## Datasets

### Subtask A
* monolingual
    * train
    * dev
* multilingual
    * train
    * dev

### Subtask B
* train
* dev

### Subtask C
* train
* dev

## Pre-training

### Subtask A
* **monolingual**
    * Continue pre-training one BERT on subtaskA.monolingual.train
    * Continue pre-training one BERT on subtaskA.monolingual.train + subtaskB.train
    * Continue pre-training one BERT for each source from subtaskA.monolingual.train
    * Continue pre-training one BERT for each source from subtaskA.monolingual.train + subtaskB.train
* **multilingual**
    * Continue pre-training one BERT-multilingual on subtaskA.multilingual.train

## Fine-tuning

### Text preprocessing
* None
* Light preprocessing (lowercase and remove punctuation)
* Heavy preprocessing (lowercase, remove punctuation, some typo fixing etc.)

### Text tokenization
* **Truncation strategies:**
    * head only: keep only first X tokens from the text
    * tail only: keep only the last X tokens from the text
    * head and tail: keep only first 1/4 * X tokens and last 3/4 * X tokens from the text
* **Hierarchical strategies:**
    * Divide text into k = L / X fractions and get the representation ([CLS] hidden state) for each fraction from the BERT model. Then combine them with different strategies:
        * Max pooling
        * Mean pooling

### Features from BERT
* Get features from the BERT using different strategies:
    * Last layer
    * Last 4 layers with different combining strategies:
        * Mean
        * Max
        * Concatenation
    * First 4 layers with different combining strategies:
        * Mean
        * Max
        * Concatenation
    * All Layers + concatenation

## Optimizer

* https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e#6196
