import numpy as np
import flair
import flair.data
import pandas as pd
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nlpaug.util import Action
import nlpaug.flow as nafc
import os
import nltk
import csv
from Text_generator import gen_list_of_exempt_words


def append_to_df(df, section, val):
    df = df[['text', section+'_bucket']]

    df = df.append(pd.read_csv('upsample_txt/'+section+val+'.csv'))
    df =df.append(pd.read_csv('up_copy_txt/'+section+val+'.csv'))
    return df

def load_df_to_sentences(df: pd.DataFrame, label_type):
    sentences = []
    for index, row in df.iterrows():
        sentence = flair.data.Sentence(row['text'])
        label = str(row['satisfaction_bucket'])  # must be a string, it's a classification
        sentence.add_label(label_type, label, 1.0)
        sentences.append(sentence)
    return sentences

def upsample(df, key, section):
    df_majority = df[df[key] == 3]
    new_df = df_majority.copy(deep=True)
    for targ in [1,5]:
        df_minority = df[df[key] == targ]
        df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples = df_majority.shape[0])
        df_minority_upsampled = upsample_in_place(df_minority_upsampled, section, targ)
        new_df = pd.concat([new_df, df_minority_upsampled, df_minority])

    return new_df


def upsample_in_place(df, section, val):
    ex_word = gen_list_of_exempt_words()

    text = df["text"].values.tolist()
    aug = naw.SynonymAug(aug_src='wordnet', stopwords=ex_word, aug_min=2, aug_max=5)
    augmented_text = aug.augment(text)
    df = pd.DataFrame(augmented_text, columns=['text'])
    buck_nam = section + '_bucket'
    df[buck_nam] = val

    return df

def main():
    df = pd.read_csv("fin_data.csv")  # i limited it to 500 for speed of training on a cpu, read all when fully training

    # Adding Upsampled data
    # print(len(df))
    # df = append_to_df(df, 'satisfaction', '5')
    # df = append_to_df(df, 'satisfaction', '1')
    # print(len(df))

    # Changing to 3 bucket approach
    df.loc[df["satisfaction_bucket"] == 2, "satisfaction_bucket"] = 1
    df.loc[df["satisfaction_bucket"] == 4, "satisfaction_bucket"] = 5

    # print(df["satisfaction_bucket"].value_counts())

    label_type = 'satisfaction'
    target = 'satisfaction_bucket'
    df = df[['text', 'satisfaction_bucket']]

    train_dev, test = train_test_split(df, test_size=0.2, stratify=df[target])
    train, dev = train_test_split(train_dev, test_size=0.2, stratify=train_dev[target])

    print(len(train), len(dev), len(test))
    df2 = pd.read_csv('reviews/bucket_reviews.csv')
    train = pd.concat([train, df2])
    print(len(train))
    print(train.value_counts())
    train = upsample(train, target, 'satisfaction')
    test = upsample(test, target, 'satisfaction')
    dev = upsample(dev, target, 'satisfaction')

    print(len(train), len(dev), len(test))


    train_sentences = load_df_to_sentences(train, label_type)
    test_sentences = load_df_to_sentences(test, label_type)
    dev_sentences = load_df_to_sentences(dev, label_type)



    # load corpus containing training, test and dev data and if CSV has a header, you can skip it
    corpus: Corpus = Corpus(train_sentences, dev_sentences, test_sentences)



    # 3. create the label dictionary
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    # 4. initialize transformer document embeddings (many models are available)
    document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)  # transformer embeddings are hard core, awesome, you can experiment

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

    # 6. initialize trainer
    trainer = ModelTrainer(classifier, corpus)

    # 7. run training with fine-tuning
    trainer.fine_tune('./test_distilbert_3_bucket',
                      learning_rate=5.0e-5,  # another good one to mess with
                      mini_batch_size=4,  # increase this, higher for a cpu, don't go above 8 on a gpu, sometimes get problem
                      max_epochs=10,  # mess with this
                      )

if __name__ == '__main__':
    main()