import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import docx2txt
import glob
import re 

import argparse

import spacy
import nltk
from nltk import tokenize

en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
nltk.download('punkt')


def parse_doc(file_path):

    with open(file_path, 'rb') as f:
        doc = docx2txt.process(f)

    word_par = [word for word in doc.rstrip().split('\n\n') if word.lower() not in stopwords] # paragraph-level 

    new_text_par = " ".join(word_par)
    new_text_par = re.sub('\s+', ' ', new_text_par)

    new_text_sent = tokenize.sent_tokenize(new_text_par)

    return new_text_sent



def show_sponsor_ents(sponsor_directory):

    # Prepare a pandas dataframe
    ent_df = pd.DataFrame(columns=['entity_type', 'keyword', 'frequency'])
    ent_dict = defaultdict(lambda: defaultdict(int))
    ent_type_lst = []
    ent_keyword_lst = []
    ent_freq_lst = []
    
    # for each file in the directory
    all_files = glob.glob(sponsor_directory + '\**\*.docx', recursive = True) # needs correction to be '/' if needed (Windows)
    for file in all_files:
        # parse docx in python-readable format
        parsed_txt = parse_doc(file)

        # apply a pre-trained spaCy model to the raw text
        for sent in parsed_txt:
          ent_sent = en(sent)

          # Count the frequency of keywords under each entity type and entity word
          if ent_sent.ents:
            for ent_word in ent_sent.ents:
              ent_dict[ent_word.label_][ent_word.text] += 1
              # print(ent_word.text+' - ' +str(ent_word.start_char) +' - '+ str(ent_word.end_char) +' - '+ent_word.label_+ ' - '+str(spacy.explain(ent_word.label_)))
          else:
            print('No named entities found.')

    # dictionary to pandas dataframe
    for each_ent in ent_dict.keys():
        each_ent_info = ent_dict[each_ent]

        for ent_keyword in each_ent_info.keys():
            ent_type_lst.append(each_ent)
            ent_keyword_lst.append(ent_keyword)
            ent_freq_lst.append(each_ent_info[ent_keyword])
      
    # Insert rows to dataframe
    ent_df['entity_type'] = ent_type_lst
    ent_df['keyword'] = ent_keyword_lst
    ent_df['frequency'] = ent_freq_lst

    # save to csv
    ent_df.to_csv(sponsor_directory + '\entity_info.csv', index=True)
    sponsor_team = sponsor_directory.split('\\')[-1]

    print(f'done saving csv with entity info for {sponsor_team}!')


def show_and_filter_ents_all(folder_directory, year, h): # input: year (Fiscal year), h (H1 or H2)

    # Concatenate all entity infos from each sponsor team's 'entity_info.csv'

    sponsors_dir = glob.glob(folder_directory + '\*\entity_info.csv', recursive=True)

    df = pd.read_csv(sponsors_dir[0])
    df['team'] = sponsors_dir[0].split('\\')[-2]

    for sponsor_file in sponsors_dir[1:]:
      new = pd.read_csv(sponsor_file)
      new['team'] = sponsor_file.split('\\')[-2]

      df = pd.concat([df, new], ignore_index=True)

    df['year'] = year
    df['H'] = h

    # Uncomment if you want to save the merged entity info all across projects at year and H to csv file!
    # df.to_csv(folder_directory + f'/{year}_{h}_entity.csv', index=False)

    # Filter out non-related entities
    
    df_cleaned = df.copy()
    
    # (1) non-related entity types (based on spaCy)
    non_related_entity = ['LOC', 'FAC', 'QUANTITY', 'GPE', 'PERSON', 'CARDINAL', 'TIME', 
                          'DATE', 'NORP', 'MONEY', 'ORDINAL', 'PERCENT'] # add more if you find new ones
    df_cleaned = df_cleaned.loc[~df_cleaned.entity_type.isin(non_related_entity)]

    # (2) only AI-related keywords
    with open('..\data\ai_glossary.txt', 'r') as f:
      ai_related_word = [line.rstrip().lower() for line in f]

    joined_ai_related_word = '|'.join(ai_related_word)
    df_cleaned = df_cleaned.loc[df_cleaned.keyword.str.lower().str.contains(joined_ai_related_word)]
    
    # Save the cleaned dataframe to csv format
    df_cleaned = df_cleaned.reset_index(drop=True)
    df_cleaned.to_csv(folder_directory + f'\cleaned_{year}_{h}_entity.csv', index=False)

    return df_cleaned


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sponsor', type=str, required=True, help="directory path for sponsor team")
    parser.add_argument('--year', required=True, help='Fiscal Year')
    parser.add_argument('--h', required=True, help='H1 or H2')
    args = parser.parse_args()

    # show named entities from the input sponsor team
    show_sponsor_ents(args.sponsor)
    
    # Show and save named entities from the entire fiscal year and h semester, all across sponsoring teams
    top_folder_before_sponsor = args.sponsor.split('\\')[-2]
    show_and_filter_ents_all(top_folder_before_sponsor,args.year, str(args.h))

    # Show and save named entities all across years and semesters -> named `all_years_merged_entity.csv`
    pd.concat((pd.read_csv(f) for f in glob.glob('..\data\**\cleaned*.csv')), ignore_index=True).to_csv('../data/all_years_merged_entity.csv', index=False)
