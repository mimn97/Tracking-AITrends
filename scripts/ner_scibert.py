import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import docx2txt
import glob
import re 
import argparse

import nltk
nltk.download('punkt')
from nltk import tokenize

import spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline

from ner_spacy import parse_doc




parser = argparse.ArgumentParser()
parser.add_argument('--proposal', type=str, required=True, help="directory path for proposal")
args = parser.parse_args()

# Load the finetuned SciBERT model and tokenizer

s_model = AutoModelForTokenClassification.from_pretrained('..\model\scibert_ner_ai.model')
s_tokenizer = AutoTokenizer.from_pretrained('..\model\scibert_ner_ai.model')

# Test on the proposal

parsed_proposal = parse_doc(args.proposal) # fill with the path to any proposal
nlp = pipeline("ner", model=s_model, tokenizer=s_tokenizer, aggregation_strategy="simple")

word_lst = []
for p in parsed_proposal:
  result = nlp(p)
  print(p)