# Introduction 

Code Repository 'Tracking AI' by Minhwa Lee

# Getting Started

## 	Installation process

- For installing neccessary packages, please command `pip install -r requirements.txt`. Please note that this is written on Python 3.8+ and torch with CUDA. 

## AI/ML/DS keywords
- In the file `data/ai_glossary.txt`, please place AI/ML/DS keywords. This will be the words that will be used for filtering out in the spaCy's NER results. 

## Running scripts
- Please clean the folder name if redundant. The best way is to follow the structure of renaming the folder as I already did in the repository.

### (1) spaCy 
- For testing Spacy's NER tagging model for each sponsoring team, here's the example command line: 

    `python ner_spacy.py --sponsor [path to the sponsor team] --year [fiscal year] --h [H1 or H2]`
    
    For example, it could be `python ner_spacy.py --sponsor '..\data\FY21H1\Sponsor Proposals\AI Platform' --year 2021 --h H1`, for the AI Platform team on FY21 H1. 

- After calling `ner_spacy.py` with all sponsoring teams in each fiscal & H folder (e.g., FY21H1, FY22 H1, FY22 H2), then you will see that in the `data` folder, there will be a csv file `all_years_merged_entity.csv`. Please download this file and upload it to Power BI for making word clouds. 

### (2) SciBERT
- First, run the script in the command: `python train_scibert.py`. This script will train a pre-trained SciBERT model on the CrossNER's AI domain datasets, for the downstream purpose of NER tasks.

- Then, run the script `python ner_scibert.py --proposal [path to any proposal]`. Then, it will show which entities from the input proposal have been extracted by our finetuned SciBERT model. 


Please let me know if there are some bugs when you run the scripts. 