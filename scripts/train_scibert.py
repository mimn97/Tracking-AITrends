import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import docx2txt
import glob
import re 
import spacy
import nltk
nltk.download('punkt')
from nltk import tokenize

en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words


from datasets import Dataset, load_metric

import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification


def read_crossner(data_path):
    """
    Aggregates words into sentences, for efficiently batching later while model development. 

    data_path: the path to CrossNER dataset (train or dev)
    """

    data = pd.read_csv(data_path, delimiter='\t', header=None)
    data = data.rename(columns={0:'word', 1:'entity'})

    entire_word_lst = []
    entire_entity_lst = []

    word_lst = []
    entity_lst = []

    for i in range(len(data)):
        if data['word'][i] != '.':
            word_lst.append(data['word'][i].lower())
            entity_lst.append(data['entity'][i])
        else:
            word_lst.append(data['word'][i].lower())
            entity_lst.append(data['entity'][i])

            entire_word_lst.append(word_lst)
            entire_entity_lst.append(entity_lst)
            word_lst = []
            entity_lst = []


    data_cleaned = pd.DataFrame({'word':entire_word_lst, 'entity': entire_entity_lst})
    return data_cleaned


def tokenize_and_labels(examples):

    label_all_tokens = True
    tokenized_inputs = tokenizer(examples["word"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["entity"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # elif label[word_idx] == '0':
            #     label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(ai_label_encoding[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(ai_label_encoding[label[word_idx]] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ai_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ai_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }




if __name__ == '__main__':

    # Read CrossNER's Train/Dev datasets

    ai_train_path = '..\data\CrossNER\ner_data\ai\train.txt'
    ai_dev_path = '..\data\CrossNER\ner_data\ai\dev.txt'

    ai_train = read_crossner(ai_train_path)
    ai_dev = read_crossner(ai_dev_path)

    # Labels of AI entities

    ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", 
                "I-algorithm", "B-researcher", "I-researcher", "B-metrics", "I-metrics", 
                "B-programlang", "I-programlang", "B-conference", "I-conference", 
                "B-university", "I-university", "B-country", "I-country", "B-person", "I-person", 
                "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]

    ai_label_encoding = {"O":0, "B-field":1, "I-field":2, "B-task":3, "I-task":4, "B-product":5, "I-product":6, 
                    "B-algorithm":7, "I-algorithm":8, "B-researcher":9, "I-researcher":10, "B-metrics":11, "I-metrics":12, 
                    "B-programlang":13, "I-programlang":14, "B-conference":15, "I-conference":16, 
                    "B-university":17, "I-university":18, "B-country":19, "I-country":20, "B-person":21, "I-person":22, 
                    "B-organisation":23, "I-organisation":24, "B-location":25, "I-location":26, "B-misc":27, "I-misc":28}

    ai_id2label = {v: k for k, v in ai_label_encoding.items()}


    # Convert to Huggingface dataset format (for batching)

    train_hf = Dataset.from_pandas(ai_train)
    dev_hf = Dataset.from_pandas(ai_dev)

    # Model Setup 
    task = 'ner'
    scibert_model_checkpoint = 'allenai/scibert_scivocab_uncased' # SciBert pre-trained on scientific papers
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16

    tokenizer = AutoTokenizer.from_pretrained(scibert_model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(scibert_model_checkpoint, 
                                                        num_labels=len(ai_labels), id2label=ai_id2label)

    train_tokenized_datasets = train_hf.map(tokenize_and_labels, batched=True)
    dev_tokenized_datasets = dev_hf.map(tokenize_and_labels, batched=True)

    # Model Training Arguments

    scibert_model_name = scibert_model_checkpoint.split("/")[-1]

    args = TrainingArguments(
            f"..\model\{scibert_model_name}-finetuned-{task}",
            evaluation_strategy = "epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            seed=42,
            num_train_epochs=15,
            weight_decay=0.001, 
            load_best_model_at_end=True,
            save_strategy = 'epoch'
            )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval") # Seqeval is the popular metric for NER tasks.

    trainer = Trainer(
                        model,
                        args,
                        train_dataset=train_tokenized_datasets,
                        eval_dataset=dev_tokenized_datasets,
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics
                    )

    # Training 

    trainer.train()

    # Evaluating the best model

    trainer.evaluate()

    # Save the best model (lowest loss on the validation)

    trainer.save_model('..\model\scibert_ner_ai.model')




    
    


