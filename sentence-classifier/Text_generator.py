import string

import pandas as pd
import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nlpaug.util import Action
import nlpaug.flow as nafc
import os
import nltk
import csv

# nltk.download('averaged_perceptron_tagger')
os.environ["MODEL_DIR"] = 'model/'


def gen_list_of_exempt_words():
    df2 = pd.read_csv("model/unigram_freq.csv")
    word_list = df2["word"]
    ex_word = word_list[0:100].values.tolist()
    cap = []
    for word in ex_word:
        cap.append(string.capwords(word))

    # Makes list of top 100 words + themselves capitalized, + the upper and lower case alphabet
    ex_word = ex_word + cap + list(string.ascii_lowercase) + list(string.ascii_uppercase)
    print("len of exempt words", len(ex_word))
    return ex_word


def main(dataframe_filepath: str, section, val):
    df = pd.read_csv(dataframe_filepath)
    buck_nam = section + '_bucket'
    mini_df = df.loc[df[buck_nam].isin([int(val)])]
    print("Size", len(mini_df))

    ex_word = gen_list_of_exempt_words()

    text = mini_df["text"].values.tolist()
    text += text
    text += text
    text += text

    print("initializing augmenter")
    # aug = naw.SynonymAug(aug_src='ppdb', model_path='model/ppdb-2.0-tldr', stopwords=ex_word, aug_min=2)
    aug = naw.SynonymAug(aug_src='wordnet', stopwords=ex_word, aug_min=2, aug_max=5)
    print("augmenter initialized")
    augmented_text = aug.augment(text)
    print("Finished augmenting")
    print(len(augmented_text))

    write_to_file('upsample_txt/', section, val, augmented_text)


def upsample_basic(path, section, val):
    df = pd.read_csv(path)
    buck_nam = section + '_bucket'
    mini_df = df.loc[df[buck_nam].isin([int(val)])]
    print("Size", len(mini_df))
    text = mini_df["text"].values.tolist()
    text += text
    text += text
    # text += text
    print(len(text))

    write_to_file('up_copy_txt/', section, val, text)





def write_to_file(dir, section, val, augmented_text):
    filename = dir + section + val + '.csv'
    buck_col = section+"_bucket"

    with open(filename, mode='w', encoding='utf-8', newline='') as up_file:
        fieldnames = [buck_col, 'text']
        f_writer = csv.writer(up_file)
        f_writer.writerow(fieldnames)
        for line in augmented_text:
            f_writer.writerow([val, line])




if __name__ == "__main__":
    upsample_basic('fin_data.csv', 'satisfaction', '1')
    main('fin_data.csv', 'satisfaction', '1')
